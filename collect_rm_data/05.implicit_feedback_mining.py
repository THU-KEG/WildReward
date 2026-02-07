import json
import re
import numpy as np
from collections import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ================= 配置区 =================
LABEL_KEY = "label"
MAX_LOOKAHEAD = 2       # 向后看2轮
SIM_THRESHOLD = 0.5     # 相似度阈值 (0.4-0.6通常是合适区间)
# =========================================
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 加载轻量级 Embedding 模型 (第一次运行会自动下载，约 80MB)
print("📥 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
# 或者 'paraphrase-multilingual-MiniLM-L12-v2' 如果中文很多建议用这个

def extract_label(turn_data):
    """提取 Label"""
    val = turn_data.get(LABEL_KEY)
    if isinstance(val, int): return val
    if isinstance(val, str):
        if val.isdigit(): return int(val)
        match = re.search(r'\[\[(\d)\]\]', val)
        if match: return int(match.group(1))
    return None

def extract_query(turn_data):
    """提取用户的 Query"""
    # 假设你的数据结构里有 'content' 或者是 'instruction'
    # WildChat 通常结构: turns[i]['instruction'] 或 turns[i]['content'] (如果是user role)
    # 这里你需要根据实际 json 结构调整
    if 'instruction' in turn_data:
        return turn_data['instruction']
    
    # 如果是 OpenAI 格式 [{"role": "user", "content": "..."}]
    # 这里假设 merge 后的 turns 列表里存的是这种结构
    if 'content' in turn_data:
        return turn_data['content']
        
    return ""

def check_topic_similarity(q1, q2):
    """计算两个 Query 的语义相似度"""
    if not q1 or not q2: return 0.0
    embeddings = model.encode([q1, q2], convert_to_tensor=True, show_progress_bar=False)
    sim = util.cos_sim(embeddings[0], embeddings[1])
    return sim.item()

def analyze_with_topic_gate(file_path, output_path):
    print(f"🚀 Starting Semantic Analysis on {file_path}")
    print(f"   Window={MAX_LOOKAHEAD}, Sim_Threshold={SIM_THRESHOLD}")
    
    stats = {
        "valid_hindsight_pos": 0, # 3 -> 5 (同话题)
        "valid_hindsight_neg": 0, # 3 -> 1/2 (同话题)
        "blocked_by_topic": 0,    # 本来想回溯，但因为话题变了被拦截的次数
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        # 为了演示速度，这里只读一部分，正式跑去掉 [:1000]
        lines = f.readlines()

    added_data = []
        
    for line in tqdm(lines):
        if not line.strip(): continue
        conv = json.loads(line)
        turns = conv['turns']
        # import pdb; pdb.set_trace()
        labels = [t["label"] for t in turns]
        queries = [t["user_query"]["content"] for t in turns] # 提取每轮的 User Query
        
        for i in range(len(turns)):
            curr_label = labels[i]
            
            # 只处理 Category 3
            if curr_label != 3: continue
            
            # 向后看窗口
            for k in range(1, MAX_LOOKAHEAD + 1):
                future_idx = i + k
                if future_idx >= len(turns): break
                
                future_label = labels[future_idx]
                if future_label is None: continue
                
                # 只有当未来是极好(5)或极坏(1/2)时，我们才费劲去算相似度
                is_potential_pos = (future_label >= 4)
                is_potential_neg = (future_label == 1)
                
                if is_potential_pos or is_potential_neg:
                    # === 核心：Topic Check ===
                    curr_q = queries[i]
                    future_q = queries[future_idx]
                    
                    similarity = check_topic_similarity(curr_q, future_q)
                    
                    if similarity < SIM_THRESHOLD:
                        # 话题变了，拦截！
                        stats["blocked_by_topic"] += 1
                        continue # 跳过这一轮回溯
                    
                    # 话题一致，允许回溯
                    if is_potential_pos:
                        turns[i]["label"] = future_label
                        history = []
                        for _j in range(i):
                            history.extend(turns[_j]["messages"])
                        turns[i]["history"] = history
                        added_data.append(turns[i])
                        stats["valid_hindsight_pos"] += 1
                        # 这里你可以记录一下 id，后面用来做训练数据
                        
                    if is_potential_neg:
                        stats["valid_hindsight_neg"] += 1
                        # 这是一个隐形地雷，记下来！
                    
                    # 只要找到了最近的一个未来信号，就不再往更远的看（可选策略）
                    # break 

    print("\n" + "="*40)
    print("📊 SEMANTIC HINDSIGHT REPORT")
    print("="*40)
    print(f"✅ Valid Positive Hindsight (3->5, Same Topic): {stats['valid_hindsight_pos']}")
    print(f"⚠️ Valid Negative Hindsight (3->1/2, Same Topic): {stats['valid_hindsight_neg']}")
    print(f"🛡️ Blocked by Topic Switch (Interventions):      {stats['blocked_by_topic']}")
    
    total_valid = stats['valid_hindsight_pos'] + stats['valid_hindsight_neg']
    if total_valid + stats['blocked_by_topic'] > 0:
        block_rate = stats['blocked_by_topic'] / (total_valid + stats['blocked_by_topic']) * 100
        print(f"📉 Topic Switch Rate in Hindsight Candidates: {block_rate:.1f}%")
        print("   (这意味着如果不加 Topic Check，你的数据会有这么多噪声！)")
    
    with open(output_path, "w") as f:
        count = 0
        for item in added_data:
            count += 1
            f.write(json.dumps(item, ensure_ascii=False)+"\n")
        print(count)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Implicit feedback mining with topic-aware hindsight"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input merged conversations JSONL file"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output added_data JSONL file"
    )
    parser.add_argument(
        "--cuda_device", type=str, default="0",
        help="CUDA device ID (default: 0)"
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # 运行
    analyze_with_topic_gate(args.input, args.output)