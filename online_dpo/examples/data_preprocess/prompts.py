import re
import os
import json
import datasets
import random
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def load_dataset(data_paths):
    def _load_dataset(data_path):
        data = []
        with open(data_path) as f:
            for line in tqdm(f.readlines()):
                item = json.loads(line.strip())
                data.append({
                    "id": "NA",
                    "prompt": item["prompt"] if isinstance(item, dict) else item.strip()
                })
        return data
    
    data = []
    for data_path, ratio in data_paths.items():
        _data = _load_dataset(data_path)
        random.shuffle(_data)
        cnt = int(len(_data)*ratio)
        data.extend(_data[:cnt])
    
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--local_dir', default='~/data/general_domain')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_paths = {
        args.data_path: 1.0
    }
 
    data_source = "remote_wild_rm"
    data_list = load_dataset(data_paths)
    dataset = datasets.Dataset.from_list(data_list)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('prompt')

            data = {
                "data_source": data_source,
                "prompt": [       
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "if",
                "reward_model": {
                    "style": "rm",
                    "ground_truth": "NA"
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
