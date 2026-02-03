# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from verl import DataProto
from verl.utils.reward_score import gsm8k, math, multiply, countdown, kk, halueval, hotpot
import torch
import nltk
import re
import json
import numpy as np
import io
import contextlib
import os
# Global counter for training step logging (print every 10 steps)
_training_step_counter = 0
def _select_rm_score_fn(data_source):
    if data_source == 'GSM8K':
        return gsm8k.compute_score
    elif data_source == 'MATH':
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    elif "kk" in data_source:
        return kk.compute_score
    elif data_source == 'HaluEval':
        return halueval.compute_score
    elif data_source in ['hotpot', '2wikimultihop', 'musique']:
        return hotpot.compute_score
    else:
        raise NotImplementedError
def _select_rm_score_fn_batch(data_source):
    if data_source in ['hotpot', '2wikimultihop', 'musique_ans', 'musique']:
        return hotpot.compute_scores_batch
    else:
        # Fallback to single computation for other data sources
        return None
class NaiveRewardManager:
    """The reward manager.
    """
    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
    def __call__(self, data: DataProto):
        # Delegate to module-level implementation to keep code structure stable
        return _naive_reward_manager_call(self, data)
def _strict_str_bool(value) -> bool:
    """Strictly accept bool or 'true'/'false' (case-insensitive).
    - bool -> returned as-is
    - str  -> only 'true'/'false' accepted
    - others -> ValueError
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s == 'true':
            return True
        if s == 'false':
            return False
        raise ValueError(f"answerable must be 'true' or 'false', got: {value}")
    raise ValueError(f"answerable must be a bool or string 'true'/'false', got: {type(value)}")
def _naive_reward_manager_call(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        global _training_step_counter
        _training_step_counter += 1
        # Always disable verbose evaluation in hotpot.py to reduce clutter
        # We'll do our own compact logging here
        os.environ['VERBOSE_EVAL'] = 'false'
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        answer_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        base_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)  # Original judge result
        # Check if we can use batch processing
        # Can be disabled via environment variable for debugging or if server can't handle concurrent requests
        enable_batch_processing = os.environ.get('ENABLE_BATCH_ANSWER_PROCESSING', 'true').lower() == 'true'
        data_source = data[0].non_tensor_batch['data_source']
        compute_scores_batch_fn = _select_rm_score_fn_batch(data_source)
        if enable_batch_processing and compute_scores_batch_fn:
            # Batch processing logic
            sequences_strs = []
            response_texts = []
            ground_truths = []
            documents_list = []
            answer_aliases_list = []
            answerable_list = []
            response_lengths = []
            for i in range(len(data)):
                data_item = data[i]
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                response_lengths.append(valid_response_length)
                valid_response_ids = response_ids[:valid_response_length]
                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=False)
                sequences_str = re.sub(r'\n{2,}', '\n', sequences_str).strip()
                response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                sequences_strs.append(sequences_str)
                response_texts.append(response_text)
                ground_truths.append(data_item.non_tensor_batch['answer'])
                documents_list.append(data_item.non_tensor_batch.get('documents', '[]'))
                # Extract answer_aliases from extra_info
                extra_info = data_item.non_tensor_batch.get('extra_info', {})
                # Ensure dict and strict answerable
                if isinstance(extra_info, str):
                    try:
                        extra_info = json.loads(extra_info)
                    except Exception:
                        extra_info = {}
                if not isinstance(extra_info, dict):
                    extra_info = {}
                aliases = extra_info.get('answer_aliases', []) if isinstance(extra_info, dict) else []
                answer_aliases_list.append(aliases)
                # answerable flag (must be 'true'/'false')
                raw_flag = extra_info['answerable']
                try:
                    ans_flag = _strict_str_bool(raw_flag)
                except ValueError:
                    print(f"[RM][Error] invalid answerable in batch: value={raw_flag} type={type(raw_flag)}; index={i}; extra_info={extra_info}")
                    raise
                answerable_list.append(ans_flag)
            scores = compute_scores_batch_fn(
                solution_strs=sequences_strs, 
                ground_truths=ground_truths,
                answer_aliases_list=answer_aliases_list,
                answerable_list=answerable_list
            )
            # Print header for this batch
            print(f"\n{'='*80}")
            print(f"[Training Step {_training_step_counter}] Batch Evaluation (batch_size={len(scores)})")
            print(f"{'='*80}")
            for i, (format_score, answer_score, base_reward) in enumerate(scores):
                total_score = answer_score
                reward_tensor[i, response_lengths[i] - 1] = total_score
                format_reward_tensor[i] = format_score
                answer_reward_tensor[i] = answer_score
                base_reward_tensor[i] = base_reward  # Store original judge result
                # Output info for each sample
                answerable_flag = answerable_list[i]
                data_source = data[i].non_tensor_batch.get('data_source', 'unknown')
                # Extract ground truth information
                try:
                    ground_truth_list = []
                    # Get main answer
                    main_answer = data[i].non_tensor_batch.get('answer', None)
                    if main_answer:
                        ground_truth_list.append(main_answer)
                    # Get answer aliases from extra_info
                    extra_info = data[i].non_tensor_batch.get('extra_info', {})
                    if isinstance(extra_info, dict):
                        answer_aliases = extra_info.get('answer_aliases', [])
                        if isinstance(answer_aliases, list):
                            ground_truth_list.extend(answer_aliases)
                    # Format ground truth for display (truncate if too many)
                    if len(ground_truth_list) > 0:
                        # Truncate each item if too long
                        ground_truth_display = []
                        for gt in ground_truth_list[:5]:  # Max 5 items
                            gt_str = str(gt)
                            if len(gt_str) > 50:
                                gt_str = gt_str[:47] + "..."
                            ground_truth_display.append(gt_str)
                        ground_truth_str = str(ground_truth_display)
                    else:
                        ground_truth_str = "[]"
                except Exception as e:
                    ground_truth_str = f"[ERROR: {str(e)}]"
                # Extract model response (answer part only)
                try:
                    # Use decoded response tokens only to avoid matching example tags in the prompt
                    response_text = response_texts[i]
                    # Optionally, if templates leak markers, strip anything before the last assistant marker
                    for marker in ["<|im_start|>assistant", "Assistant:", "<|start_header_id|>assistant", "<｜Assistant｜>"]:
                        if marker in response_text:
                            response_text = response_text.split(marker)[-1]
                            break
                    # Try to extract just the answer tag content
                    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        # Truncate if too long
                        if len(answer_content) > 100:
                            answer_content = answer_content[:100] + "..."
                    else:
                        # If no answer tag found, show the first 100 chars of response
                        answer_content = f"[NO TAG] {response_text[:80].strip()}..."
                except Exception as e:
                    answer_content = f"[ERROR: {str(e)}]"
                # Determine score category
                if total_score >= 0.999:
                    score_label = "✓ CORRECT"
                elif total_score <= -0.999:
                    score_label = "✗ INCORRECT"
                else:
                    score_label = "◯ MISS"
                # Color-coded output based on answerable
                answerable_str = f"ans={answerable_flag}"
                print(f"  [{i+1:2d}/{len(scores)}] {data_source:15s} | {answerable_str:12s} | "
                      f"score={total_score:+.1f} {score_label:12s} | ans=\"{answer_content}\" | "
                      f"gt={ground_truth_str}")
            print(f"{'='*80}\n")
            return reward_tensor, format_reward_tensor, answer_reward_tensor, base_reward_tensor
        # --- Fallback to single processing ---
        # Print header for this batch
        print(f"\n{'='*80}")
        print(f"[Training Step {_training_step_counter}] Batch Evaluation - Single Processing (batch_size={len(data)})")
        print(f"{'='*80}")
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=False)
            sequences_str = re.sub(r'\n{2,}', '\n', sequences_str).strip()
            # Also decode response-only text to extract answer safely
            response_only_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            ground_truth = data_item.non_tensor_batch['answer']
            data_source = data_item.non_tensor_batch['data_source']
            documents = data_item.non_tensor_batch.get('documents', '[]')
            # Extract answer_aliases from extra_info
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            if isinstance(extra_info, str):
                try:
                    extra_info = json.loads(extra_info)
                except Exception:
                    extra_info = {}
            if not isinstance(extra_info, dict):
                extra_info = {}
            answer_aliases = extra_info.get('answer_aliases', []) if isinstance(extra_info, dict) else []
            raw_flag = extra_info['answerable']
            try:
                answerable_flag = _strict_str_bool(raw_flag)
            except ValueError:
                print(f"[RM][Error] invalid answerable: value={raw_flag} type={type(raw_flag)}; index={i}; extra_info={extra_info}")
                raise
            compute_score_fn = _select_rm_score_fn(data_source)
            # Always suppress verbose output from compute_score_fn
            _sink = io.StringIO()
            with contextlib.redirect_stdout(_sink):
                # Only pass answer_aliases for data sources that support it
                if data_source in ['hotpot', '2wikimultihop', 'musique_ans', 'musique']:
                    format_score, answer_score, base_reward = compute_score_fn(
                        response=sequences_str,
                        ground_truth=ground_truth,
                        documents=documents,
                        answerable=answerable_flag,
                        answer_aliases=answer_aliases,
                        enable_postprocessing=True,
                    )
                else:
                    format_score, answer_score, base_reward = compute_score_fn(
                        response=sequences_str,
                        ground_truth=ground_truth,
                        documents=documents,
                        answerable=True,
                        enable_postprocessing=True,
                    )
            total_score = answer_score
            reward_tensor[i, valid_response_length - 1] = total_score
            format_reward_tensor[i] = format_score
            answer_reward_tensor[i] = answer_score
            base_reward_tensor[i] = base_reward  # Store original judge result
            # Prepare ground truth display
            try:
                ground_truth_list = []
                # Get main answer
                if ground_truth:
                    ground_truth_list.append(ground_truth)
                # Add answer aliases
                if isinstance(answer_aliases, list):
                    ground_truth_list.extend(answer_aliases)
                # Format ground truth for display (truncate if too many)
                if len(ground_truth_list) > 0:
                    # Truncate each item if too long
                    ground_truth_display = []
                    for gt in ground_truth_list[:5]:  # Max 5 items
                        gt_str = str(gt)
                        if len(gt_str) > 50:
                            gt_str = gt_str[:47] + "..."
                        ground_truth_display.append(gt_str)
                    ground_truth_str = str(ground_truth_display)
                else:
                    ground_truth_str = "[]"
            except Exception as e:
                ground_truth_str = f"[ERROR: {str(e)}]"
            # Extract model response (answer part only)
            try:
                # Use response-only text so we never match example tags from the prompt
                response_text = response_only_text
                # Try to extract just the answer tag content
                answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # Truncate if too long
                    if len(answer_content) > 100:
                        answer_content = answer_content[:100] + "..."
                else:
                    # If no answer tag found, show the first 100 chars of response
                    answer_content = f"[NO TAG] {response_text[:80].strip()}..."
            except Exception as e:
                answer_content = f"[ERROR: {str(e)}]"
            # Determine score category
            if total_score >= 0.999:
                score_label = "✓ CORRECT"
            elif total_score <= -0.999:
                score_label = "✗ INCORRECT"
            else:
                score_label = "◯ MISS"
            # Output info for each sample
            answerable_str = f"ans={answerable_flag}"
            print(f"  [{i+1:2d}/{len(data)}] {data_source:15s} | {answerable_str:12s} | "
                  f"score={total_score:+.1f} {score_label:12s} | ans=\"{answer_content}\" | "
                  f"gt={ground_truth_str}")
        print(f"{'='*80}\n")
        return reward_tensor, format_reward_tensor, answer_reward_tensor, base_reward_tensor