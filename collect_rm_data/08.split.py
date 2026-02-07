import os
import json
import math
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

import argparse

label_map = {1:1, 2:2, 4:3, 5:4}

def write_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item)+"\n")

def drop(item):
    # import pdb; pdb.set_trace()
    # length = len(item["messages"][1]["content"].split())
    length = len(item["text"].split())
    prob = min(0.3 + 0.4 * math.log1p(length) / math.log1p(500), 0.9)
    return prob


def build_text(item):
    """
    构造用于 LLM 打分的 Prompt。
    输入: item (包含 history, messages)
    输出: 完整的 Prompt 字符串
    """
    
    # 1. 处理 History (上下文)
    # WildChat 或常见的 dataset history 可能是 list of dicts，也可能是 string
    history_str = ""
    history_data = item["history"]
    
    for turn in history_data:
        # 兼容常见的 role/content 键名
        role = turn.get("role", "User")
        content = turn.get("content", "")
        history_str += f"{role}: {content}\n"
    
    if not history_str.strip():
        history_str = "No previous context."

    # 2. 获取当前轮次的 Query 和 Response
    query = item["messages"][0]["content"]
    response = item["messages"][1]["content"]
    # 3. 构建 Prompt
    text = f"""
# Task Description
You are an expert conversation evaluator. Your task is to judge the **User's Satisfaction** with the Assistant's response based on the conversation context.
Please rate the response on a scale of 1 to 5 integers.

# Scoring Criteria
[1] CLEARLY NEGATIVE / REJECTION
[2] CORRECTION / ERROR POINTER (Negative)
[3] NEUTRAL
[4] POSITIVE ENGAGEMENT
[5] CLEAR SATISFACTION

# Input Data
## Context (History)
{history_str}

## User Query
{query}

## Assistant Response
{response}

# Output
Based on the criteria above, please output ONLY the integer score (1, 2, 3, 4, or 5).
"""
    return text.strip()


# ================= 1. 工具函数 =================

def get_content_length(content, language):
    """
    简单的长度计算函数
    中文：按字符计算
    英文：按单词计算
    """
    if not content:
        return 0
    if language == "Chinese":
        return len(content.replace(" ", ""))  # 去除空格算字数
    else:
        return len(content.split())  # 按空格分词算词数

# ================= 2. 定义过滤规则 (正则与关键词) =================

# --- A. 正则表达式 (预编译以提高速度) ---
# 网址链接
URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
# 邮箱地址
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
# 简单的手机号 (以国内手机号为例，避免过滤掉年份等数字)
PHONE_PATTERN = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)') 

# --- B. 关键词列表 (全部小写) ---

# 1. 图像生成 (Image Gen)
KEYWORDS_IMAGE = [
    "draw", "generate an image", "create an image", "make a picture", "sketch", 
    "midjourney", "stable diffusion", "dalle", "dall-e",
    "画一张", "画个", "画一幅", "生成图片", "生成一张", "绘图", "画画"
]

# 2. 联网/搜索 (Search)
KEYWORDS_SEARCH = [
    "browse the web", "search for", "look up online", "google it", 
    "latest news", "current weather", "what is the link", 
    "搜索一下", "查一下", "上网查", "联网", "最新的新闻", "天气怎么样"
]

# 3. 身份询问 (Identity)
KEYWORDS_IDENTITY = [
    "who are you", "what is your name", "are you an ai", "introduce yourself", 
    "openai", "chatgpt", "gpt-4", "gpt-3", "anthropic", "claude", "llama",
    "你是谁", "你的名字", "你叫什么", "你的身份", "介绍你自己", "你是ai吗", "你是机器人吗"
]

# 4. 缺少上下文/文件引用 (Missing Context)
KEYWORDS_FILE = [
    "uploaded file", "attached", "this pdf", "the document", "screenshot", 
    "上传的文件", "附件", "这张图", "这个pdf", "文档里", "截图", "看图"
]

# # 5. 拒绝回答 (Refusal - 针对 Response)
# KEYWORDS_REFUSAL = [
#     "i cannot", "i can't", "i am unable to", "as an ai", "as a language model",
#     "sorry, but", "against my policy", 
#     "我无法", "我不能", "作为一个人工智能", "作为ai", "抱歉", "无法满足"
# ]

# 6. 多媒体指令 (Media)
KEYWORDS_MEDIA = [
    "play video", "watch this", "listen to", "audio file",
    "播放视频", "听这段", "看视频"
]

# ================= 3. 主处理逻辑 =================

def filter_dataset(input_file):
    data = []
    counter = Counter()
    all_lengths = Counter()
    # 统计计数器
    stats = {
        "total": 0,
        "kept": 0,
        "dropped_label": 0,         # label不合格
        "dropped_lang": 0,          # 语言不合格
        "dropped_img": 0,           # 图像生成
        "dropped_search": 0,        # 联网/网址
        "dropped_id": 0,            # 身份认知
        "dropped_file": 0,          # 文件引用
        "dropped_pii": 0,           # 隐私信息
        "dropped_refusal": 0,       # 模型拒答
        "dropped_truncated": 0,     # 回复截断
        "dropped_length": 0,        # 长度过短
        "dropped_long_turn": 0      # turn过长
    }

    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            stats["total"] += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # import pdb; pdb.set_trace()

            # --- 0. 基础字段提取 ---
            messages = item["messages"]
                
            user_query = messages[0]["content"]
            response_text = messages[1]["content"] # 假设是单轮对话，取第二个为回复
            
            # 转小写用于匹配
            query_lower = user_query.lower()
            response_lower = response_text.lower()
            
            language = item.get("language", "English") # 默认 English 防止报错

            # --- 1. Label 和 语言过滤 (最快) ---
            if item.get("label") == 3:
                stats["dropped_label"] += 1
                continue
                
            if language not in ["Chinese", "English"]:
                stats["dropped_lang"] += 1
                continue

            # --- 2. 基于 Prompt (User Query) 的关键词过滤 ---
            # 2.1 图像生成
            if any(kw in query_lower for kw in KEYWORDS_IMAGE):
                stats["dropped_img"] += 1
                continue

            # 2.2 联网与网址
            if URL_PATTERN.search(query_lower) or any(kw in query_lower for kw in KEYWORDS_SEARCH):
                stats["dropped_search"] += 1
                continue

            # 2.3 身份认知
            if any(kw in query_lower for kw in KEYWORDS_IDENTITY):
                stats["dropped_id"] += 1
                continue
            
            # 2.4 文件引用/缺少上下文
            if any(kw in query_lower for kw in KEYWORDS_FILE):
                stats["dropped_file"] += 1
                continue

            # 2.5 隐私信息 (PII)
            if EMAIL_PATTERN.search(user_query) or PHONE_PATTERN.search(user_query):
                stats["dropped_pii"] += 1
                continue
                
            # 2.6 多媒体
            if any(kw in query_lower for kw in KEYWORDS_MEDIA):
                stats["dropped_file"] += 1 # 归类到文件/媒体类
                continue

            # --- 3. 基于 Response 的质量过滤 ---

            # 3.1 拒答过滤 (Refusal)
            # 逻辑：包含拒答词 且 长度较短 (<100字符)，避免过滤掉长篇大论的合理解释
            # if len(response_text) < 100 and any(kw in response_lower for kw in KEYWORDS_REFUSAL):
                # stats["dropped_refusal"] += 1
                # continue

            # 3.2 截断过滤 (Truncation)
            # 逻辑：检查结尾字符，且检查代码块 ``` 是否成对
            # valid_endings = ['.', '!', '?', '"', "'", '`', '。', '！', '？', '”', '’', '—', '}', ']']
            # clean_resp = response_text.strip()
            
            # # A. 检查结尾标点 (如果长度够长，但没标点结尾，大概率截断)
            # if len(clean_resp) > 50 and clean_resp[-1] not in valid_endings:
            #     stats["dropped_truncated"] += 1
            #     continue
            
            # # B. 检查代码块闭合
            # if clean_resp.count("```") % 2 != 0:
            #     stats["dropped_truncated"] += 1
            #     continue

            if len(item["history"]) > 20:
                stats["dropped_long_turn"] += 1
                continue

            # --- 4. 长度过滤 (计算量最大，放最后) ---
            query_len = get_content_length(user_query, language)
            # first_msg_len = get_content_length(messages[0]["content"], language)
            response_len = get_content_length(response_text, language)

            # 长度阈值设置
            # if first_msg_len < 5:
                # stats["dropped_length"] += 1
                # continue
            
            # 回复太短通常没有学习价值
            if response_len < 10: 
                stats["dropped_length"] += 1
                continue
                
            # (可选) 如果 Prompt 极短，可能也没意义
            if query_len < 5:
                stats["dropped_length"] += 1
                continue

            # --- 5. 通过所有检查，写入 ---
            stats["kept"] += 1

            # import pdb; pdb.set_trace()
            
            counter[item["label"]] += 1
            all_lengths[item["label"]] += response_len
            # print()
            item["label"] = label_map[item["label"]]
            item["text"] = build_text(item)
            new_item = {
                "id": item["id"],
                "history": item["history"],
                "text": item["text"],
                "messages": item["messages"],
                "user_feedback": item["user_query"],
                "label": item["label"]
            }
            data.append(new_item)

    # --- 6. 打印统计报告 ---
    print("="*30)
    print("Filter Statistics Report")
    print("="*30)
    print(f"Total processed:     {stats['total']}")
    print(f"✅ Kept:              {stats['kept']} ({stats['kept']/stats['total']:.2%})")
    print("-" * 20)
    print(f"❌ Dropped (Details):")
    print(f"   Label/Score:      {stats['dropped_label']}")
    print(f"   Language:         {stats['dropped_lang']}")
    print(f"   Image Gen:        {stats['dropped_img']}")
    print(f"   Search/URL:       {stats['dropped_search']}")
    print(f"   Identity:         {stats['dropped_id']}")
    print(f"   File/Media:       {stats['dropped_file']}")
    print(f"   PII (Privacy):    {stats['dropped_pii']}")
    print(f"   Refusal:          {stats['dropped_refusal']}")
    print(f"   Truncated:        {stats['dropped_truncated']}")
    print(f"   Length Short:     {stats['dropped_length']}")
    print(f"   Long Turn:     {stats['dropped_long_turn']}")
    print("="*30)

    for key in counter:
        all_lengths[key] /= counter[key]
    # for key in counter:
        # counter[key] /= len(data)

    print(counter, len(data), all_lengths)
    return data


def main(input1, input2, train_output, test_output, test_count=5000):
    """Filter and split data into train/test sets"""
    data1 = filter_dataset(input1)
    data2 = filter_dataset(input2)
    data = data1 + data2
    random.shuffle(data)

    counter = Counter()
    for item in data:
        counter[item["label"]] += 1

    for key in counter:
        counter[key] /= len(data)
    print("Label distribution:", counter)
    print("Total data:", len(data))

    train = data[:-test_count]
    test = data[-test_count:]

    # Create output directories if needed
    Path(train_output).parent.mkdir(parents=True, exist_ok=True)
    Path(test_output).parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(train, train_output)
    write_jsonl(test, test_output)

    print(f"✅ Train set saved to: {train_output} ({len(train)} samples)")
    print(f"✅ Test set saved to: {test_output} ({len(test)} samples)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter and split data into train/test sets"
    )
    parser.add_argument(
        "--input1", type=str, required=True,
        help="Path to first input JSONL file (final.jsonl)"
    )
    parser.add_argument(
        "--input2", type=str, required=True,
        help="Path to second input JSONL file (added_data.jsonl)"
    )
    parser.add_argument(
        "--train_output", type=str, required=True,
        help="Path to output train JSONL file"
    )
    parser.add_argument(
        "--test_output", type=str, required=True,
        help="Path to output test JSONL file"
    )
    parser.add_argument(
        "--test_count", type=int, default=5000,
        help="Number of samples for test set (default: 5000)"
    )

    args = parser.parse_args()

    main(args.input1, args.input2, args.train_output, args.test_output, args.test_count)
