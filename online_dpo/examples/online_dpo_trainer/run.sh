cd ../data_preprocess
python prompts.py --data_path <your path>
cd ../online_dpo_trainer
export REWARD_MODEL_ENDPOINT="<your endpoint>"
bash run_llama3_8b.sh