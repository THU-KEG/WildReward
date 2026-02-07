import json
import re
from tqdm import tqdm
from collections import defaultdict


def merge_jsonl(file1, output_path):
    # 1. 读取两个jsonl文件
    data = []
    for path in [file1]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    # 2. 按 conversation_id 分组
    conversations = defaultdict(list)
    for item in tqdm(data):
        conv_id = item["id"].split("_")[0]
        turn_id = int(item["id"].split("_")[1].replace("turn", ""))
        item["turn_id"] = turn_id
        del item["history"]
        conversations[conv_id].append(item)

    print(len(conversations))
    # 3. 每个对话按轮次排序并提取内容
    merged_data = []
    for conv_id, turns in tqdm(conversations.items()):
        turns_sorted = sorted(turns, key=lambda x: x["turn_id"])
        conv_entry = {
            "conversation_id": conv_id,
            "turns": turns_sorted
        }

        # for t in turns_sorted:
        #     # 优先使用原始字段，否则从prompt中提取
        #     conv_entry["turns"].append(turn_tobj)

        merged_data.append(conv_entry)
        # import pdb; pdb.set_trace()

    # 4. 输出为新的 JSONL 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in merged_data:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    print(f"✅ Merged {len(merged_data)} conversations saved to {output_path}")


# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge JSONL file into conversation format"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output merged JSONL file"
    )

    args = parser.parse_args()

    merge_jsonl(
        args.input,
        args.output
    )
