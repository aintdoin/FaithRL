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
import re
import string
from collections import Counter
from typing import Dict, Tuple, Optional
def extract_answer(solution_str, method='strict'):
    assert method in ['strict', 'flexible']
    if method == 'strict':
        # this also tests the formatting of the model
        final_answer = solution_str.replace(',', '').replace('$', '')
    elif method == 'flexible':
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer
def validate_model_answer(answer_text: str, expected_answer: str):
    """Parses model's answer text into status dictionary.
    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_answer: Text extracted from data
    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    print(f"\n[Answer Validation]")
    print(f"  Expected answer: {expected_answer}")
    print(f"  Predicted answer: {answer_text}")
    final_answer = extract_answer(answer_text)
    return 1.0 if final_answer == expected_answer else 0.0
def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    Args:
        processed_str: Processed response string from the model
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Format Validation]")
    validation_passed = True
    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }
    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        print(f"  {tag_str}: count={count}, position={pos}")
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False
    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
            positions['think_end'] > positions['answer_start'] or
            positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    return validation_passed
def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    Args:
        solution_str: Raw response string from the language model
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<｜Assistant｜>" in solution_str:
        processed_str = solution_str.split("<｜Assistant｜>", 1)[1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    else:
        print("  [Error] Failed to locate model response header")
        return None, solution_str
    # Extract reasoning and final answer using XML-style tags
    reasoning_pattern = r'<think>(.*?)</think>'
    matches = list(re.finditer(reasoning_pattern, processed_str, re.DOTALL))
    if not matches:
        print("\n  [Error] No valid reasoning text found")
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        print("\n  [Error] No valid answer text found")
        answer_text = None
    else:
        answer_text = matches[-1].group(1).strip()
    return answer_text, processed_str
def compute_score(solution_str: str, ground_truth: str):
    """Computes comprehensive score for model response.
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "=" * 80)
    print(" Processing New Sample ".center(80, '='))
    # Parse ground truth data
    print(f"\n[Ground Truth]  {ground_truth}")
    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")
    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = 0 if format_correct else -2
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")
    # Validate answer content
    if format_correct and answer_text:
        answer_score = validate_model_answer(answer_text, ground_truth)
        print(f"  Answer score: {answer_score}")
    else:
        answer_score = 0
        error = "missing answer" if format_correct else "format errors"
        print(f"\n[Answer Validation] Skipped due to {error} (Answer score: {answer_score})")
    return format_score, answer_score