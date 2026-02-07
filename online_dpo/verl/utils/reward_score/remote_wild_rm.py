import requests
import json
import math
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed


api_url = None


# ======================
#   Build Chat Text
# ======================
def build_text(query, response, history_str=""):
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


# ======================
#   Parse Solution
# ======================
def parse_solution(text):
    return text.split("</think>")[-1].strip()


# ======================
#   One batch request
# ======================
def batch_request(text_list):
    headers = {"Content-Type": "application/json"}
    data = {"query": text_list}

    if api_url is None:
        raise ValueError("You need set the api_url for your remote reward model first.")

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=600)
        result = response.json()
        return result["rewards"]
    except Exception as e:
        print(f"❌ 批量请求失败: {e}", response, response.json())
        return [2] * len(text_list)


# ======================
#   Worker for threads
# ======================
def process_batch(batch, start_index):
    """
    batch: list of {"instruction":..., "answer":...}
    start_index: 用于合并结果
    """
    texts = [build_text(item["instruction"], item["answer"]) for item in batch]
    rewards = batch_request(texts)
    return start_index, rewards


# ======================
#   Multi-thread + batch
# ======================
def local_request_mt(data_list, batch_size=64, max_threads=16):
    futures = []
    results = [None] * len(data_list)

    with ThreadPoolExecutor(max_threads) as executor:
        for idx in range(0, len(data_list), batch_size):
            batch = data_list[idx: idx + batch_size]
            futures.append(
                executor.submit(process_batch, batch, idx)
            )

        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing (multi-thread batch)"):
            start_index, rewards = f.result()
            for i, r in enumerate(rewards):
                results[start_index + i] = r

    return results


# ======================
#   Compute Score
# ======================
def compute_score(prompt_str, solution_str, ground_truth):
    solution_str = [parse_solution(s) for s in solution_str]

    data = []
    for i in range(len(solution_str)):
        data.append({
            "instruction": prompt_str[i],
            "answer": solution_str[i],
        })

    results = local_request_mt(
        data,
        batch_size=32,     
        max_threads=32    
    )
    print(results)
    return results


# ======================
#   Main
# ======================
if __name__ == "__main__":
    compute_score(
        ["ok" * 1024] * 16,
        ["ok" * 1024] * 16,
        ["NA<sepsepsep>NA"] * 16
    )


