import os
import json
from tqdm import trange
import pandas as pd
from pathlib import Path
import random


def to_message(item):
    """确保每条消息是 {'role':..., 'content':...} 格式"""
    if isinstance(item, dict) and "role" in item and "content" in item:
        return {"role": item["role"], "content": item["content"]}
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        return {"role": item[0], "content": item[1]}
    elif isinstance(item, str):
        return {"role": "user", "content": item}
    else:
        return {"role": "unknown", "content": str(item)}


def reformat(panda_table):
    data = []
    for i in trange(len(panda_table)):
        item = panda_table.iloc[i]
        conversation = item["conversation"]
        # 确保完整的user/assistant对齐
        if len(conversation) < 4:
            continue

        # 一轮 = 用户 + 助手，共2条消息
        num_turns = len(conversation) // 2

        # 从第2轮开始（N>1）
        for N in range(1, num_turns):
            # if random.random() < 0.9:
                # continue
            # 取第 N-1 和第 N 轮
            start_idx = (N - 1) * 2
            end_idx = N * 2  # 不包含end_idx
            # target_messages = conversation[start_idx:end_idx]

            # 历史是前 N-1 轮的全部对话
            history_messages = [to_message(m) for m in conversation[: (N - 1) * 2]]
            target_messages = [to_message(m) for m in conversation[start_idx:end_idx]]
            user_query = to_message(conversation[end_idx])

            if item["language"] not in ["Chinese", "English"]:
                continue
            
            assert target_messages[0]["role"] == "user"
            assert target_messages[-1]["role"] == "assistant"
            
            new_item = {
                "id": f"{item['conversation_hash']}_turn{N-1}",
                "model": item["model"],
                "turn": N-1,
                "history": history_messages,
                "messages": target_messages,
                "user_query": user_query,
                "language": item.get("language", "unknown"),
            }
            data.append(new_item)
    return data


def extract_data(input_path, save_path):
    df = pd.read_parquet(input_path)
    file_name = os.path.basename(input_path).replace(".parquet", ".jsonl")

    data = reformat(df)
    print(f"{file_name}: {len(data)} items")
    with open(os.path.join(save_path, file_name), "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess WildChat parquet data to JSONL format"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to directory containing WildChat parquet files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./processed_data_all",
        help="Path to output directory for processed JSONL files"
    )

    args = parser.parse_args()

    dir = args.input_dir
    save_path = Path(args.output_dir)
    save_path.mkdir(exist_ok=True)

    for file in os.listdir(dir):
        if not file.endswith(".parquet"):
            continue
        input_path = os.path.join(dir, file)
        extract_data(input_path, save_path)
