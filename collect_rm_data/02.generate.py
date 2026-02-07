import os
import sys
import json
import datasets
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

from build_model import APIModel, vLLMModel, LocalAPIModel
from collections import defaultdict

def save_results(save_path, outputs):
    def save_jsonlines(save_path, outputs):
        with open(save_path, "w") as f:
            for output in outputs:
                f.write(json.dumps(output, ensure_ascii=False)+"\n")
    if save_path.endswith(".jsonl"):
        save_jsonlines(save_path, outputs)
    elif save_path.endswith(".json"):
        json.dump(outputs, open(save_path, "w"), ensure_ascii=False, indent=4)
    else:
        raise NotImplementedError()


def merge_api_results(raw_data):
    data = defaultdict(list)
    raw_items = dict()
    for item in raw_data:
        data[item["prompt"]].append(item["response"])
        raw_items[item["prompt"]] = item
    final_data = []
    for key, values in data.items():
        item = raw_items[key]
        item.pop("response", None)
        item["responses"] = list(values)
        final_data.append(item)
    return final_data

def load_data(input_file, n=1):
    def load_jsonlines(input_file):
        data = []
        with open(input_file) as f:
            for line in f.readlines():
                data.append(json.loads(line.strip()))
        return data
    if "wildchat" in input_file.lower():
        dataset = []
        raw_dataset = []
        with open(input_file) as f:
            for line in f.readlines():
                item = json.loads(line.strip())
                for _ in range(n):
                    raw_dataset.append(item)
                    dataset.append([item["messages"][0]])
        return dataset, raw_dataset
    if "infinity-instruct" in input_file.lower():
        dataset = []
        raw_dataset = []
        with open(input_file) as f:
            for line in f.readlines()[:100000]:
                item = json.loads(line.strip())
                for _ in range(n):
                    raw_dataset.append(item)
                    dataset.append(
                        [{"role": "user", "content": item["prompt"]}]
                    )
        return dataset, raw_dataset

    if "polycritic" in input_file.lower():
        dataset = []
        raw_dataset = []
        with open(input_file) as f:
            for line in f.readlines():
                item = json.loads(line.strip())
                for _ in range(n):
                    raw_dataset.append(item)
                    dataset.append(
                        [{"role": "user", "content": item["prompt"]}]
                    )
        return dataset, raw_dataset
    
    if "generated_prompts" in input_file.lower():
        dataset = []
        raw_dataset = []
        with open(input_file) as f:
            for line in f.readlines():
                item = json.loads(line.strip())
                for _ in range(n):
                    raw_dataset.append(item)
                    dataset.append(
                        [{"role": "user", "content": item["prompt"]}]
                    )
        return dataset, raw_dataset


    if "rm-bench" in input_file.lower():
        dataset = []
        raw_dataset = []
        with open(input_file) as f:
            for line in f.readlines():
                item = json.loads(line.strip())
                for _ in range(n):
                    raw_dataset.append(item)
                    dataset.append(
                        [{"role": "user", "content": item["prompt"]}]
                    )
        return dataset, raw_dataset


def main(args, input_file, model, n):
    if args.api_model:
        dataset, raw_dataset = load_data(input_file, n=n)
        def update_progress_bar(done, total):
            # Simple text-based progress bar
            progress = int(50 * done / total)  # Calculate progress (50 chars width)
            sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
            sys.stdout.flush()
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Map 'my_function' across the vector, executing in parallel using threads
            # results = list(executor.map(get_judgement, dataset))

            # Progress bar version
            results = [None] * len(dataset)  # Preallocate results list
            done_tasks = 0  # Counter for completed tasks

            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                # Submit all tasks and hold their futures in a list
                future_to_index = {executor.submit(lambda x: model.generate_chat(x, max_tokens=args.max_tokens, temperature=args.temperature), x): i for i, x in enumerate(dataset)}

                # As tasks complete, update progress and store results in the original order
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    done_tasks += 1
                    update_progress_bar(done_tasks, len(dataset))

            # Print newline after progress bar
            print()

            final_outputs = []
            for item, output in zip(raw_dataset, results):
                if "prompt" not in item:
                    item["prompt"] = item["messages"][0]["content"]
                item["response"] = output
                item["generator"] = args.model_name_or_path.split("/")[-1]
                final_outputs.append(item)
            return final_outputs
    else:
        dataset, raw_dataset = load_data(input_file, n=1)
        outputs = model.generate(
            dataset,
            {
                "n": n,
                "temperature": args.temperature,
                "top_p": 0.95,
                "max_tokens": args.max_tokens
            }
        )
        final_outputs = []
        for item, output in zip(raw_dataset, outputs):
            if "prompt" not in item:
                item["prompt"] = item["messages"][0]["content"]
            # item["responses"] = output
            item["generated"] = output[0].split("</think>")[-1].strip()
            item["generator"] = args.model_name_or_path.split("/")[-1]
            final_outputs.append(item)
        
        return final_outputs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--api_model", action="store_true")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--num_threads", default=64, type=int)
    parser.add_argument("--input_key", default=None)
    parser.add_argument("--output_key", default=None)
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--save_name", default=None)
    parser.add_argument("--api_url", default="http://172.18.197.81:8000/v1/chat/completions", type=str)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--max_tokens", default=256, type=int)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    if args.api_model:
        model = APIModel(base_url=os.environ["BASE_URL"], model_name=args.model_name_or_path, api_key=os.environ["API_KEY"])
        # model = LocalAPIModel(args.api_url, args.model_name_or_path)
        # model = ZAIModel(model_name=args.model_name_or_path, api_key=os.environ["OPENAI_API_KEY"])
    else:
        model = vLLMModel(args.model_name_or_path, num_gpus=args.n_gpus, vllm_gpu_util=0.9)
    outputs = main(args, args.input_file, model, args.n)
    # if args.api_model: # merge
    #     outputs = merge_api_results(outputs)

    if args.save_json:
        if args.save_name is not None:
            save_results(os.path.join(save_dir, args.save_name), outputs)
        else:
            save_results(os.path.join(save_dir, f"{args.n}_responses.json"), outputs)
    else:
        if args.save_name is not None:
            save_results(os.path.join(save_dir, args.save_name), outputs)
        else:
            save_results(os.path.join(save_dir, f"{args.n}_responses.jsonl"), outputs)