""" Preprocess dataset for knights and knaves logic task """
import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import random
def make_prefix(dp, template_type):
    question = dp['question']
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the assistant solves it. 
        The assistant first thinks about the reasoning process in the mind and then provides the user with the final 
        answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, 
        respectively. For example, <think> reasoning process here </think> <answer> answer here </answer>. The Assistant 
        shows the reasoning process within <think> </think> tags, and ONLY return the FINAL ANSWER within <answer> </answer> 
        tags. For example: <answer> 12 </answer>.\n\nUser: {question}\nAssistant: Let me solve this step by step. <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. Given a question, you need to first think about 
        the reasoning process in the mind and then provide the final answer. The reasoning process and answer are enclosed 
        within <think> </think> and <answer> </answer> tags, respectively. For example, <think> reasoning process 
        here </think> <answer> answer here </answer>. You must show the reasoning process within <think> </think> tags, 
        and ONLY return the FINAL ANSWER within <answer> </answer> tags. For example: <answer> 12 </answer>.<|im_end|>\n<|im_start|>user\n{question}\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    elif template_type == 'llama-instruct':
        prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. 
        Given a question, you need to first think about the reasoning process in the mind and then provide the final 
        answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, 
        respectively. For example, <think> reasoning process here </think> <answer> answer here </answer>. You must show 
        the reasoning process within <think> </think> tags, and ONLY return the FINAL ANSWER within <answer> </answer> tags. For example: <answer> 12 </answer>.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet me solve this step by step.\n<think>"""
    return prefix
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/math-500/qwen-instruct')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='/home/project/11004114/rl/data/MATH-500/final_test.json')
    parser.add_argument('--train_size', type=int, default=5000)
    parser.add_argument('--test_size', type=int, default=5000)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    args = parser.parse_args()
    data_source = 'MATH'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                yield json.loads(line)
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(len(raw_dataset))
    # assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    # test_dataset = raw_dataset.select(range(TRAIN_SIZE))
    # train_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    test_dataset = raw_dataset
    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "factuality",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example["answer"]
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    # train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)
    # train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)