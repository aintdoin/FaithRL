#!/usr/bin/env python3
"""
OOD Inference Script
Supports multiple .jsonl files, independent evaluation, and OOD-specific metrics.
"""
import argparse
import json
import os
import sys
import re
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
# Add parent directory to path to import verl modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verl.utils.reward_score.answer_postprocessor import get_postprocessor
from verl.utils.dataset.system_prompts import wrap_prompt_with_system
def parse_args():
    parser = argparse.ArgumentParser(description="Run OOD inference")
    parser.add_argument("--test-files", type=str, required=True, help="List of .jsonl files (e.g. \"['file1.jsonl', 'file2.jsonl']\")")
    parser.add_argument("--output-dir", type=str, required=True, help="Base output directory")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    return parser.parse_args()
def normalize_answer(text):
    if not text:
        return ""
    # Strip whitespace, lowercase, and remove trailing period
    return str(text).strip().lower().rstrip('.')
def extract_answer_content(text):
    if not text:
        return ""
    # Extract content between <answer> and </answer>
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()
def extract_think_content(text):
    if not text:
        return ""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""
def is_idk(text):
    markers = {"i don't know", "i dont know", "insufficient information", "unknown"}
    return normalize_answer(text) in markers
def split_reasoning_steps(reasoning_str):
    # Regex from verl/workers/fsdp_workers.py
    enum_pattern = re.compile(r'(?m)^(\s*)(\d+)\.(\s*)')
    markers = list(enum_pattern.finditer(reasoning_str))
    steps = []
    if not markers:
        # If no numbered steps found, treat whole thing as one step if not empty
        if reasoning_str.strip():
            steps.append(reasoning_str.strip())
        return steps
    for i, m in enumerate(markers):
        # Start of content for this step (after the number and dot)
        content_start = m.end() 
        # End of content is start of next marker or end of string
        if i + 1 < len(markers):
            content_end = markers[i+1].start()
        else:
            content_end = len(reasoning_str)
        step_content = reasoning_str[content_start:content_end].strip()
        if step_content:
            steps.append(step_content)
    return steps
def evaluate_step_validity(step_text, evidences, postprocessor):
    """
    Uses LLM Judge to evaluate if a step is supported by evidences.
    Returns 1.0 (valid) or 0.0 (invalid/unsupported).
    """
    if not postprocessor.use_judge_api:
        return 0.0
    # Format evidences
    evd_list = evidences if isinstance(evidences, list) else [evidences]
    evd_text = '\n'.join([f"- {str(e)}" for e in evd_list if e])
    # Construct prompt (based on verl/workers/fsdp_workers.py)
    prompt_lines = []
    prompt_lines.append("You are a strict reasoning consistency judge. Decide if the reasoning segment is FULLY SUPPORTED by the provided evidences.")
    prompt_lines.append("Rules:")
    prompt_lines.append("1) Output only one digit: 1 if the segment contains meaningful reasoning AND is strictly supported by the evidences; 0 otherwise.")
    prompt_lines.append("2) Give 0 if the segment adds NO new information, is just a plan/re-statement, or lacks specific details (e.g., 'We should review the list...').")
    prompt_lines.append("3) Give 1 ONLY if the segment's key assertion semantically matches or is directly inferred from an evidence.")
    prompt_lines.append("4) Base the decision strictly on the evidences; ignore world knowledge.")
    prompt_lines.append("5) Do not provide explanations.")
    if evd_text:
        prompt_lines.append("")
        prompt_lines.append(f"Evidences:\n{evd_text}")
    prompt_lines.append("")
    prompt_lines.append(f"Reasoning Segment:\n{step_text}")
    prompt_lines.append("")
    prompt_lines.append("Output (only 0 or 1):")
    judge_prompt = '\n'.join(prompt_lines)
    # Call Judge API
    try:
        result_text = postprocessor._call_judge_api(judge_prompt)
        s = result_text.strip()
        if s == '1':
            return 1.0
        if s == '0':
            return 0.0
        # Fallback parsing
        m = re.match(r'^\s*([01])', s)
        if m:
            return 1.0 if m.group(1) == '1' else 0.0
    except Exception as e:
        print(f"Judge API error: {e}")
    return 0.0
def main():
    args = parse_args()
    # Parse file list
    import ast
    try:
        test_files = ast.literal_eval(args.test_files)
        if not isinstance(test_files, list):
            test_files = [args.test_files]
    except:
        test_files = [args.test_files]
    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens
    )
    postprocessor = get_postprocessor()
    print(f"LLM Judge Enabled: {postprocessor.use_judge_api}")
    # Clear the combined output file if it exists
    combined_output_file = os.path.join(args.output_dir, "combined_results.jsonl")
    if os.path.exists(combined_output_file):
        os.remove(combined_output_file)
    for file_path in test_files:
        print(f"\n{'='*80}")
        print(f"Processing: {file_path}")
        print(f"{'='*80}")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        # Load data
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        if args.num_samples != -1:
            data = data[:args.num_samples]
        print(f"Loaded {len(data)} samples")
        # Prepare prompts
        prompts = []
        # Determine prompt type based on file path
        # GPQA uses default OOD prompt (with options A/B/C/D)
        # Math datasets (GSM8k, MATH500) use the new MATH prompt
        if "GPQA" in file_path:
            os.environ['SYSTEM_PROMPT_TYPE'] = 'ood'
            print(f"Using SYSTEM_PROMPT_TYPE=ood for {file_path}")
        else:
            os.environ['SYSTEM_PROMPT_TYPE'] = 'math'
            print(f"Using SYSTEM_PROMPT_TYPE=math for {file_path}")
        for item in data:
            question = item.get('question', '')
            # OOD requirement: Only System Prompt + Question
            full_prompt = wrap_prompt_with_system(question, model_template=os.environ.get('MODEL_TEMPLATE', 'qwen'))
            prompts.append(full_prompt)
        # Inference
        outputs = llm.generate(prompts, sampling_params)
        # Stats
        stats = {
            'total': 0,
            'correct': 0,
            'miss': 0,
            'hallucination': 0,
            # Step stats [valid_steps, total_steps]
            'steps_correct': [0, 0],
            'steps_miss': [0, 0],
            'steps_hallucination': [0, 0],
            'steps_total': [0, 0]
        }
        results = []
        print("Evaluating results...")
        for i, (item, output) in enumerate(tqdm(zip(data, outputs), total=len(data))):
            generated_text = output.outputs[0].text
            # Extract components
            answer_content = extract_answer_content(generated_text)
            think_content = extract_think_content(generated_text)
            ground_truth = str(item.get('answer', '')).strip()
            evidences = item.get('evidences', [])
            # 1. Classification
            is_correct = False
            is_miss = False
            is_hallucination = False
            norm_pred = normalize_answer(answer_content)
            # Handle possible list of answers
            gt_raw = item.get('answer', '')
            if isinstance(gt_raw, list):
                possible_answers = [normalize_answer(a) for a in gt_raw]
                if norm_pred in possible_answers:
                    is_correct = True
            else:
                norm_gt = normalize_answer(gt_raw)
                if norm_pred == norm_gt:
                    is_correct = True
            if is_correct:
                stats['correct'] += 1
            elif is_idk(answer_content):
                is_miss = True
                stats['miss'] += 1
            else:
                is_hallucination = True
                stats['hallucination'] += 1
            stats['total'] += 1
            # 2. Process Evaluation (Steps)
            steps = split_reasoning_steps(think_content)
            valid_steps = 0
            total_steps = len(steps)
            if total_steps > 0 and postprocessor.use_judge_api:
                for step in steps:
                    score = evaluate_step_validity(step, evidences, postprocessor)
                    if score > 0.5: # 1.0
                        valid_steps += 1
            # Update step stats
            if total_steps > 0:
                stats['steps_total'][0] += valid_steps
                stats['steps_total'][1] += total_steps
                if is_correct:
                    stats['steps_correct'][0] += valid_steps
                    stats['steps_correct'][1] += total_steps
                elif is_miss:
                    stats['steps_miss'][0] += valid_steps
                    stats['steps_miss'][1] += total_steps
                elif is_hallucination:
                    stats['steps_hallucination'][0] += valid_steps
                    stats['steps_hallucination'][1] += total_steps
            # Record result
            result_item = {
                'data_source': item.get('data_source', 'unknown'),
                'question': item.get('question'),
                'ground_truth': ground_truth,
                'model_output': generated_text,
                'prediction': answer_content,
                'status': 'correct' if is_correct else 'miss' if is_miss else 'hallucination',
                'num_steps': total_steps,
                'valid_steps': valid_steps,
                'step_validity_rate': (valid_steps / total_steps) if total_steps > 0 else 0.0
            }
            results.append(result_item)
        # Calculate Rates
        total = stats['total'] if stats['total'] > 0 else 1
        acc = stats['correct'] / total
        miss_rate = stats['miss'] / total
        hal_rate = stats['hallucination'] / total
        def safe_div(num, den):
            return num / den if den > 0 else 0.0
        step_validity_total = safe_div(stats['steps_total'][0], stats['steps_total'][1])
        step_validity_correct = safe_div(stats['steps_correct'][0], stats['steps_correct'][1])
        step_validity_miss = safe_div(stats['steps_miss'][0], stats['steps_miss'][1])
        step_validity_hal = safe_div(stats['steps_hallucination'][0], stats['steps_hallucination'][1])
        print(f"\nResults for {os.path.basename(file_path)}:")
        print(f"Total Samples: {stats['total']}")
        print(f"Accuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")
        print(f"Miss Rate (IDK): {miss_rate:.2%} ({stats['miss']}/{stats['total']})")
        print(f"Hallucination Rate: {hal_rate:.2%} ({stats['hallucination']}/{stats['total']})")
        print("-" * 40)
        print("Process Validity (Step-wise):")
        print(f"Overall Step Validity: {step_validity_total:.2%} ({stats['steps_total'][0]}/{stats['steps_total'][1]})")
        print(f"  - Correct Samples: {step_validity_correct:.2%} ({stats['steps_correct'][0]}/{stats['steps_correct'][1]})")
        print(f"  - Miss Samples:    {step_validity_miss:.2%} ({stats['steps_miss'][0]}/{stats['steps_miss'][1]})")
        print(f"  - Hallucination:   {step_validity_hal:.2%} ({stats['steps_hallucination'][0]}/{stats['steps_hallucination'][1]})")
        # Save results to combined file
        combined_output_file = os.path.join(args.output_dir, "combined_results.jsonl")
        os.makedirs(os.path.dirname(combined_output_file), exist_ok=True)
        with open(combined_output_file, 'a', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
        # Save summary
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        summary_file = os.path.join(args.output_dir, f"{base_name}_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"File: {file_path}\n")
            f.write(f"Total: {stats['total']}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Miss Rate: {miss_rate:.4f}\n")
            f.write(f"Hallucination Rate: {hal_rate:.4f}\n")
            f.write(f"Step Validity Overall: {step_validity_total:.4f}\n")
            f.write(f"Step Validity Correct: {step_validity_correct:.4f}\n")
            f.write(f"Step Validity Miss: {step_validity_miss:.4f}\n")
            f.write(f"Step Validity Hallucination: {step_validity_hal:.4f}\n")
    if postprocessor:
        postprocessor.shutdown()
if __name__ == "__main__":
    main()