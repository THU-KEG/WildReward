import os
import re
import json

# 定义拒绝回答的关键词列表
KEYWORDS_REFUSAL = [
    "i cannot", "i can't", "i am unable to", "as an ai", "as a language model",
    "sorry, but", "against my policy", "against my programming",
    "我无法", "我不能", "作为一个人工智能", "作为ai", "抱歉", "对不起", "无法满足"
]

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.strip():
                data.append(json.loads(line))
    return data

def parse(output_text):
    if not output_text:
        return 0
    match = re.search(r"\[\[(\d)\]\]\s*(.*)", output_text.strip())
    try:
        category = int(match.group(1))
    except:
        category = 0
    return category

def is_refusal(content):
    """检查内容是否包含拒答关键词"""
    if not content:
        return False
    content_lower = content.lower()
    if len(content) < 300: 
        for kw in KEYWORDS_REFUSAL:
            if kw in content_lower:
                return True
    return False

def merge(input_file, reann_file, output_file):
    print("Loading data...")
    data = load_jsonl(input_file)
    data_reann = load_jsonl(reann_file)

    # 1. 获取白名单 ID (safe_ids)
    ids_1 = []
    ids_5 = []
    count = 0
    na_count = 0

    for item in data_reann:
        if item.get("response") is None:
            na_count += 1
            continue
        label = parse(item["response"])
        if label == 0:
            ids_1.append(item["id"])
            count += 1
        if label == 1:
            ids_5.append(item["id"])
            count += 1

    # 转为集合，查询速度 O(1)
    safe_ids_set_1 = set(ids_1)
    safe_ids_set_5 = set(ids_5)
    print(f"Safe IDs count: {count}, NA count: {na_count}", len(safe_ids_set_1), len(safe_ids_set_5))

    # 2. 过滤并写入
    kept_count = 0
    filtered_count = 0

    with open(output_file, "w", encoding='utf-8') as f:
        for item in data:
            item_id = item.get("id")

            if item["label"] == 3:
                continue

            # 逻辑分支 A: 在白名单中 (data_reann中label为1的)
            if item_id in safe_ids_set_1:
                item["label"] = 1
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept_count += 1
                continue # 直接处理下一个，跳过 refusal 检查

            if item_id in safe_ids_set_5:
                item["label"] = 5
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept_count += 1
                continue # 直接处理下一个，跳过 refusal 检查

            # 否则保留
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept_count += 1

    print(f"Processing complete.")
    print(f"Total kept: {kept_count}")
    print(f"Filtered (Refusal): {filtered_count}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge refusal validation results into final dataset"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input merged.jsonl file"
    )
    parser.add_argument(
        "--reann", type=str, required=True,
        help="Path to safe_judge_annotated.jsonl file"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output final.jsonl file"
    )

    args = parser.parse_args()

    merge(args.input, args.reann, args.output)