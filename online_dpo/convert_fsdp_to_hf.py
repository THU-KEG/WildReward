from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from glob import glob
from collections import defaultdict
import argparse


def convert(fsdp_checkpoint_path, huggingface_model_path, output_path):
    state_dict = defaultdict(list)

    world_size = 8
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath, weights_only=False)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    #for filepath in glob(f'{fsdp_checkpoint_path}/model_*.pt'):
    #    part_state_dict = torch.load(filepath)
    #    model.load_state_dict(part_state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fsdp_checkpoint_path", help="path to FSDP shards, e.g. global_step_xxx/actor")
    parser.add_argument("hf_reference_path", help="path to reference HF model (config+tokenizer)")
    parser.add_argument("output_path", help="path to save HuggingFace model")

    args = parser.parse_args()

    convert(args.fsdp_checkpoint_path, args.hf_reference_path, args.output_path)