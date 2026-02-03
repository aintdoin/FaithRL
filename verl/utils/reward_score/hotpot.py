import re
import json
import string
import os
import random
from collections import Counter
from typing import Dict, Tuple, Optional
from .answer_postprocessor import get_postprocessor
def _extract_answer_content(text: str) -> str:
    """
    Extract content between <answer> and </answer> tags.
    Returns empty string if tags not found.
    """
    if not text:
        return ""
    # Find <answer> and </answer> tags (case insensitive)
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
def _count_tokens_simple(text: str) -> int:
    """
    Simple token counting: split by whitespace and punctuation.
    This is a rough approximation but fast and deterministic.
    For GPT-like tokenizers, this tends to underestimate slightly,
    which is conservative for penalties.
    """
    if not text:
        return 0
    # Split by whitespace and common punctuation
    # This gives a rough token count
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return len(tokens)
def _calculate_length_penalty(answer_text: str, max_free_tokens: int = 10, penalty_per_token: float = 0.1, max_penalty: float = 2.0) -> float:
    """
    Calculate penalty for answer length exceeding threshold.
    Args:
        answer_text: Full response text (may contain <answer> tags)
        max_free_tokens: Number of tokens allowed without penalty (default: 10)
        penalty_per_token: Penalty per extra token (default: 0.1)
        max_penalty: Maximum total penalty (default: 2.0)
    Returns:
        Penalty value (0 if within limit, positive value otherwise, capped at max_penalty)
    """
    # Extract content between <answer> tags
    answer_content = _extract_answer_content(answer_text)
    # If no answer tags found, use the full text (fallback)
    if not answer_content:
        answer_content = answer_text
    # Count tokens
    token_count = _count_tokens_simple(answer_content)
    # Calculate penalty
    if token_count <= max_free_tokens:
        return 0.0
    excess_tokens = token_count - max_free_tokens
    penalty = excess_tokens * penalty_per_token
    # Cap at maximum penalty
    penalty = min(penalty, max_penalty)
    return penalty
def _calculate_repetition_penalty(text: str, max_penalty: float = 2.0) -> float:
    """
    Calculate penalty for repeated lines in reasoning/answer.
    Ignores leading numbers/bullets to catch numbered list repetitions.
    """
    if not text:
        return 0.0
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return 0.0
    # Regex to remove leading numbers/bullets (e.g. "1. ", "- ", "1) ")
    pattern = re.compile(r'^(\d+[\.\)]|\-|\*)\s*')
    normalized_lines = []
    for line in lines:
        # Remove leading marker
        cleaned = pattern.sub('', line).strip()
        if len(cleaned) > 10:  # Only consider lines with significant content
            normalized_lines.append(cleaned)
    if not normalized_lines:
        return 0.0
    counts = Counter(normalized_lines)
    penalty = 0.0
    for count in counts.values():
        if count > 1:
            # Penalize each repetition (0.5 per extra occurrence)
            penalty += (count - 1) * 0.5
    return min(penalty, max_penalty)
def _is_idk_response(text: Optional[str]) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    patterns = [
        "i don't know",
        "i dont know",
        "i am not sure",
        "i'm not sure",
        "cannot answer",
        "can't answer",
        "cannot determine",
        "can't determine",
        "insufficient information",
        "not enough information",
        "unknown",
        "no sufficient information",
    ]
    for p in patterns:
        if p in t:
            return True
    return False
def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s
def normalize_answer(s):
    # Convert to string first to handle int/float inputs
    if not isinstance(s, str):
        s = str(s)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)
    def lower(text):
        return text.lower()
    def replace_underscore(text):
        return text.replace("_", " ")
    def normalize_synonyms(text):
        """Normalize common synonyms for boolean answers"""
        text = text.strip()
        # Yes synonyms
        if text in ["correct", "true", "right", "affirmative", "yeah", "yep", "yup"]:
            return "yes"
        # No synonyms
        if text in ["incorrect", "false", "wrong", "negative", "nope", "nah"]:
            return "no"
        return text
    def normalize_numbers(text):
        """Remove thousand separators from numbers"""
        # Match numbers with commas: 1,234 or 1,234.56
        text = re.sub(r'(\d),(\d)', r'\1\2', text)
        return text
    def normalize_dates(text):
        """Normalize common date format variations"""
        # "until" <-> "to": "1969 until 1974" <-> "1969 to 1974"
        text = re.sub(r'\buntil\b', 'to', text)
        # Handle date ranges with hyphen: "1969-1974" -> "1969 to 1974"
        # Match pattern: number-number (with word boundaries or spaces)
        text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 to \2', text)
        return text
    # Apply all normalizations in order
    normalized = lower(replace_underscore(s))
    normalized = normalize_dates(normalized)
    normalized = normalize_numbers(normalized)
    normalized = remove_punc(normalized)
    normalized = remove_articles(normalized)
    normalized = white_space_fix(normalized)
    normalized = normalize_synonyms(normalized)
    return normalized
def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))
    ZERO_METRIC = (0, 0, 0)
    special_answers = ["yes", "no", "no answer"]
    if normalized_prediction in special_answers or normalized_ground_truth in special_answers:
        if normalized_prediction in normalized_ground_truth.split() or normalized_ground_truth in normalized_prediction.split():
            return 1.0, 1.0, 1.0
        else:
            return ZERO_METRIC
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall
def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(bool_mapping(ground_truth))
def cover_exact_match_score_1(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split()
    ground_list = normalize_answer(bool_mapping(ground_truth)).split()
    return all(token in pre_list for token in ground_list)
def cover_exact_match_score_2(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split()
    ground_list = normalize_answer(bool_mapping(ground_truth)).split()
    for i in range(len(pre_list) - len(ground_list) + 1):
        if pre_list[i : i + len(ground_list)] == ground_list:
            return True
    pre_str = " ".join(pre_list)
    ground_str = " ".join(ground_list)
    if ground_str in pre_str:
        return True
    return False
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    if metric_fn.__name__ == "exact_match_score":
        for ground_truth in ground_truths:
            em_score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(em_score)
        return max(scores_for_ground_truths)
    elif metric_fn.__name__ == "f1_score":
        for ground_truth in ground_truths:
            f1, precision, recall = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append((f1, precision, recall))
        f1, precision, recall = max(scores_for_ground_truths, key=lambda x: x[0])
        return f1, precision, recall
    elif metric_fn.__name__ == "cover_exact_match_score_1":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    elif metric_fn.__name__ == "cover_exact_match_score_2":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    else:
        raise NotImplementedError
def compute_metrics(prediction, gold):
    em = metric_max_over_ground_truths(exact_match_score, prediction, gold)
    f1, precision, recall = metric_max_over_ground_truths(f1_score, prediction, gold)
    cover_em_1 = metric_max_over_ground_truths(cover_exact_match_score_1, prediction, gold)
    cover_em_2 = metric_max_over_ground_truths(cover_exact_match_score_2, prediction, gold)
    metrics = dict()
    metrics["em"] = float(em)
    metrics["cover_em_1"] = float(cover_em_1)
    metrics["cover_em_2"] = float(cover_em_2)
    metrics["f1"] = f1
    metrics["precision"] = precision
    metrics["recall"] = recall
    if cover_em_1:
        metrics["acc_num"] = 1
    return metrics
def _get_dynamic_idk_penalty():
    """
    Read current validation test_score and compute dynamic IDK penalty.
    Formula: min(0, max(2*test_score - 1, -0.4))
    Uses test_score = 2*n_correct + n_miss - 1, not just n_correct ratio.
    This formula creates an adaptive penalty that:
    - When test_score < 0.3: penalty = -0.4 (fixed penalty to discourage all-IDK strategy)
    - When 0.3 ≤ test_score < 0.5: penalty = 2*test_score-1 (gradually reduces from -0.4 to 0)
    - When test_score ≥ 0.5: penalty = 0 (no penalty, model is good enough to judge)
    Examples:
    - test_score=0.20: min(0, max(-0.6, -0.4)) = min(0, -0.4) = -0.4
    - test_score=0.30: min(0, max(-0.4, -0.4)) = min(0, -0.4) = -0.4
    - test_score=0.40: min(0, max(-0.2, -0.4)) = min(0, -0.2) = -0.2
    - test_score=0.50: min(0, max(0.0, -0.4)) = min(0, 0.0) = 0.0
    - test_score=0.578: min(0, max(0.156, -0.4)) = min(0, 0.156) = 0.0
    Returns:
        float: IDK penalty value (default: 0.0 if dynamic penalty disabled)
    """
    import os
    enable_dynamic_idk = os.environ.get('ENABLE_DYNAMIC_IDK_PENALTY', 'false').lower() == 'true'
    if not enable_dynamic_idk:
        # Fallback to static penalty
        return float(os.environ.get('IDK_PENALTY_ANSWERABLE', '0.0'))
    # Try to read accuracy from state file
    # The file path should match the trainer's checkpoint directory
    trainer_dir = os.environ.get('TRAINER_DEFAULT_LOCAL_DIR', 'checkpoints')
    state_file = os.path.join(trainer_dir, 'dynamic_idk_state.txt')
    default_accuracy = 0.3  # Conservative default
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                accuracy = float(f.read().strip())
        else:
            # File doesn't exist yet (first training iteration)
            accuracy = default_accuracy
    except Exception as e:
        # Fallback to default if reading fails
        accuracy = default_accuracy
    # Apply formula: min(0, max(2a-1, -0.4))
    inner = max(2.0 * accuracy - 1.0, -0.4)
    penalty = min(0.0, inner)
    return penalty
def validate_model_answer(answer_text: str, expected_answer: str, question: str = None, enable_postprocessing: bool = True, answer_aliases: list = None, answerable_flag: Optional[bool] = None):
    """Parses model's answer text into status dictionary.
    Args:
        answer_text: Text extracted from model's <answer> tags (raw, unprocessed)
        expected_answer: Text extracted from data
        question: Question text for context-aware extraction
        enable_postprocessing: Whether to apply advanced post-processing
        answer_aliases: List of acceptable answer aliases
    Returns:
        Dictionary with accuracy score (0 or 1) using LLM-as-a-Judge
    """
    # Removed verbose print to reduce log clutter
    # print(f"\n[Answer Validation]")
    import os
    use_llm_judge = os.environ.get('USE_LLM_JUDGE', 'false').lower() == 'true'
    # Combine expected_answer with answer_aliases
    all_expected_answers = []
    if isinstance(expected_answer, list):
        all_expected_answers.extend(expected_answer)
    else:
        all_expected_answers.append(expected_answer)
    # Add answer_aliases if provided
    if answer_aliases is not None and len(answer_aliases) > 0:
        all_expected_answers.extend(answer_aliases)
    # Use LLM-as-a-Judge for scoring (SIMPLIFIED: direct judging, no extraction)
    if use_llm_judge:
        postprocessor = get_postprocessor()  # Simplified - no parameters needed
        # Ensure answerable flag defaults to True when missing
        # Do NOT use bool() as it converts 'false' string to True!
        if answerable_flag is None:
            ans_flag = True
        else:
            ans_flag = answerable_flag
            if not isinstance(ans_flag, bool):
                raise ValueError(f"answerable_flag must be boolean or None, got {type(ans_flag)}: {ans_flag}")
        # Try all expected answers (including aliases) and take max score
        scores = []
        if ans_flag is True:
            for exp_ans in all_expected_answers:
                if exp_ans:  # Skip empty/None answers
                    score = postprocessor.judge_answer_correctness(
                        predicted_answer=answer_text,  # Use raw answer directly
                        ground_truth_answer=exp_ans,
                        question=question,
                        answerable=ans_flag
                    )
                    scores.append(score)
        if scores:
            accuracy = max(scores)  # 0/1 correctness from judge
        else:
            accuracy = 0.0
        # IDK heuristic
        is_idk = _is_idk_response(answer_text)
        # Map to {-1,0,1} with answerable awareness
        # CRITICAL FIX: Judge may return wrong score (e.g., parse "1." from "1. The question...")
        # We need to cross-check with IDK heuristic for robustness
        # Get dynamic or static IDK penalty for answerable questions
        idk_penalty_answerable = _get_dynamic_idk_penalty()
        if ans_flag is True:
            # For answerable questions:
            # - If Judge explicitly returns 0, respect it (IDK)
            # - If Judge returns 1 BUT text is IDK → likely Judge error, map to idk_penalty_answerable
            # - If Judge returns 1 AND text is NOT IDK → correct answer, map to 1
            # - If Judge returns -1 → incorrect answer, map to -1
            if accuracy >= 0.99:
                # Judge says "correct", but double-check IDK
                mapped = idk_penalty_answerable if is_idk else 1.0
            elif accuracy <= -0.99:
                # Judge says "incorrect"
                mapped = -1.0
            else:
                # Judge says "IDK" (0) or fallback to heuristic
                mapped = idk_penalty_answerable if is_idk else -1.0
        elif ans_flag is False:
            # For unanswerable questions: correct OR IDK both map to 1
            mapped = 1.0 if (accuracy >= 0.99 or is_idk) else -1.0
        else:
            # No answerable flag (fallback)
            mapped = float(accuracy)
        # Apply length penalty to the base reward
        # Read configuration from environment variables
        enable_length_penalty = os.environ.get('ENABLE_ANSWER_LENGTH_PENALTY', 'false').lower() == 'true'
        max_free_tokens = int(os.environ.get('ANSWER_MAX_FREE_TOKENS', '10'))
        penalty_per_token = float(os.environ.get('ANSWER_PENALTY_PER_TOKEN', '0.1'))
        min_final_reward = float(os.environ.get('ANSWER_MIN_FINAL_REWARD', '-2.0'))
        final_reward = float(mapped)
        length_penalty = 0.0
        if enable_length_penalty:
            # Calculate the maximum penalty that can be applied
            # to ensure final_reward >= min_final_reward
            # max_applicable_penalty = base_reward - min_final_reward
            max_applicable_penalty = mapped - min_final_reward
            # Calculate raw length penalty based on answer content
            raw_penalty = _calculate_length_penalty(
                answer_text=answer_text,
                max_free_tokens=max_free_tokens,
                penalty_per_token=penalty_per_token,
                max_penalty=float('inf')  # No cap here, we'll cap below
            )
            # Cap the penalty to ensure final reward doesn't go below minimum
            length_penalty = min(raw_penalty, max_applicable_penalty)
            # Apply penalty (subtract from reward)
            final_reward = mapped - length_penalty
            # Final safety check (should already be satisfied)
            final_reward = max(final_reward, min_final_reward)
        # Return metrics with final reward (backward-compatible keys)
        metrics = {
            "em": float(final_reward),
            "accuracy": float(final_reward),
            "f1": float(final_reward),  # downstream uses this as answer_score (with penalty)
            "precision": float(final_reward),
            "recall": float(final_reward),
            "cover_em_1": float(final_reward),
            "cover_em_2": float(final_reward),
            "base_reward": float(mapped)  # ALWAYS include original judge result for evaluation
        }
        # Add debug info for monitoring
        if enable_length_penalty and length_penalty > 0:
            answer_content = _extract_answer_content(answer_text)
            token_count = _count_tokens_simple(answer_content if answer_content else answer_text)
            metrics["answer_token_count"] = token_count
            metrics["length_penalty"] = float(length_penalty)
        if mapped == 1.0:
            metrics["acc_num"] = 1
        return metrics
    else:
        # Fallback to rule-based F1 (with optional extraction)
        original_answer = answer_text
        use_llm = os.environ.get('USE_LLM_ANSWER_EXTRACTION', 'false').lower() == 'true'
        if enable_postprocessing and use_llm:
            postprocessor = get_postprocessor()
            # Use first expected answer as reference
            exp_ref = all_expected_answers[0] if all_expected_answers else None
            answer_text = postprocessor.process_answer(
                answer_text, 
                expected_answer=exp_ref,
                question=question
            )
        # Compute metrics against all expected answers (including aliases)
        metrics = compute_metrics(answer_text, all_expected_answers)
        return metrics
def extract_solution(solution_str: str) -> Tuple[Optional[str], str, Optional[str]]:
    """Extracts the final answer and question from the model's response string.
    Args:
        solution_str: Raw response string from the language model
    Returns:
        Tuple containing (extracted_answer, processed_string, question_text)
    """
    # Extract question from the user/human part
    question_text = None
    question_markers = [
        ("User:", "Assistant:"),
        ("<|im_start|>user", "<|im_start|>assistant"),
        ("<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>"),
        ("Human:", "Assistant:"),
    ]
    for start_marker, end_marker in question_markers:
        if start_marker in solution_str and end_marker in solution_str:
            try:
                user_part = solution_str.split(start_marker)[-1].split(end_marker)[0]
                # Clean up question text
                question_text = user_part.strip()
                # Remove common prefixes
                for prefix in ['\n', '<|im_end|>', '\n\n']:
                    question_text = question_text.strip(prefix).strip()
                break
            except:
                pass
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.rsplit("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.rsplit("<|im_start|>assistant", 1)[1]
    elif "<｜Assistant｜>" in solution_str:
        processed_str = solution_str.rsplit("<｜Assistant｜>", 1)[1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        processed_str = solution_str.rsplit("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    elif '<think>' in solution_str and '<answer>' in solution_str:
        processed_str = solution_str
    else:
        # No markers and no tags - can't extract
        return None, solution_str, question_text
    # Extract reasoning and final answer using XML-style tags
    reasoning_pattern = r'<think>(.*?)</think>'
    matches = list(re.finditer(reasoning_pattern, processed_str, re.DOTALL))
    if not matches:
        # Removed verbose error print to reduce log clutter
        # print("\n  [Error] No valid reasoning text found")
        pass
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        # Removed verbose error print to reduce log clutter
        # print("\n  [Error] No valid answer text found")
        answer_text = None
    else:
        answer_text = matches[-1].group(1).strip()
    return answer_text, processed_str, question_text
def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    Args:
        processed_str: Processed response string from the model
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    validation_passed = True
    # Check required tags (silent)
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
        if count != expected_count:
            validation_passed = False
    # Verify tag order (silent)
    if (positions['think_start'] > positions['think_end'] or
            positions['think_end'] > positions['answer_start'] or
            positions['answer_start'] > positions['answer_end']):
        validation_passed = False
    return validation_passed
def compute_score(
    response,
    ground_truth,
    documents,
    answerable,
    language="en",
    use_cot=False,
    enable_postprocessing=True,
    answer_aliases=None,
    **kwargs,
):
    # answerable should already be a boolean from naive.py (_strict_str_bool)
    # Do NOT use bool() here as it converts 'false' string to True!
    # Just ensure it's a proper boolean type
    if not isinstance(answerable, bool):
        raise ValueError(f"answerable must be a boolean, got {type(answerable)}: {answerable}")
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except (SyntaxError, ValueError):
            ground_truth = [ground_truth]
    # Extract model answer and question (silent)
    answer_text, processed_str, question_text = extract_solution(response)
    # Validate response structure (silent)
    format_correct = validate_response_structure(processed_str)
    format_score = 0 if format_correct else -1
    # Simplified evaluation output (controlled by environment variable)
    import os
    verbose_eval = os.environ.get('VERBOSE_EVAL', 'false').lower() == 'true'
    if verbose_eval:
        print("\n" + "=" * 80)
        print(" Sample Evaluation ".center(80, '='))
        print("=" * 80)
        print(f"\n[Question]")
        print(f"  {question_text[:400] if question_text else 'N/A'}{'...' if question_text and len(question_text) > 400 else ''}")
        print(f"\n[Ground Truth Answer]")
        print(f"  {ground_truth}")
        print(f"\n[Model Response]")
        answer_match = re.search(r'<answer>(.*?)</answer>', processed_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            print(f"  <answer>: {answer_content}")
        else:
            print(f"  <answer>: [NOT FOUND]")
    # Validate answer content
    if format_correct and answer_text:
        metrics = validate_model_answer(
            answer_text, 
            ground_truth, 
            question=question_text,
            enable_postprocessing=enable_postprocessing,
            answer_aliases=answer_aliases,
            answerable_flag=answerable
        )
        answer_score = metrics["f1"]  # With penalty
        base_reward = metrics.get("base_reward", answer_score)  # Without penalty
        em_score = metrics["em"]
        # Apply repetition penalty (on the full processed string including reasoning)
        enable_repetition_penalty = os.environ.get('ENABLE_REPETITION_PENALTY', 'true').lower() == 'true'
        if enable_repetition_penalty:
            rep_penalty = _calculate_repetition_penalty(processed_str)
            if rep_penalty > 0:
                answer_score -= rep_penalty
                if verbose_eval:
                    print(f"  [Penalty] Repetition penalty: -{rep_penalty}")
        # Display Judge Evaluation Details (only if verbose)
        if verbose_eval:
            print(f"\n[LLM Judge Evaluation]")
            print(f"  Predicted:  {answer_text}")
            print(f"  Ground Truth: {ground_truth}")
            print(f"  Answerable: {answerable}")
            print(f"  Final Answer Score: {answer_score}")
    else:
        # Assign -1 for structure errors or missing answer as requested
        answer_score = -1.0
        base_reward = -1.0
        em_score = -1.0
    return format_score, answer_score, base_reward
def compute_scores_batch(solution_strs: list[str],
                         ground_truths: list[str],
                         enable_postprocessing: bool = True,
                         answer_aliases_list: list = None,
                         answerable_list: list = None):
    """Computes comprehensive scores for a batch of model responses.
    Args:
        solution_strs: List of model response strings
        ground_truths: List of ground truth answers
        enable_postprocessing: Whether to apply advanced post-processing
        answer_aliases_list: List of answer alias lists (one per sample)
        answerable_list: List of answerable flags (one per sample)
    Returns:
        List of tuples: (format_score, answer_score, base_reward)
        - format_score: Format validation score
        - answer_score: Final score with length penalty (for training)
        - base_reward: Original judge result without penalty (for evaluation)
    """
    # 1. Extract solutions and check formats
    extracted_data = [extract_solution(s) for s in solution_strs]
    answer_texts, processed_strs, question_texts = zip(*extracted_data)
    # For batch processing, we suppress intermediate prints from validation
    import io
    import contextlib
    import os
    format_results = []
    for ps in processed_strs:
        with contextlib.redirect_stdout(io.StringIO()):
            format_results.append(validate_response_structure(ps))
    format_scores = [0 if correct else -1 for correct in format_results]
    # Get length penalty settings
    enable_length_penalty = os.environ.get('ENABLE_ANSWER_LENGTH_PENALTY', 'false').lower() == 'true'
    max_free_tokens = int(os.environ.get('ANSWER_MAX_FREE_TOKENS', '10'))
    penalty_per_token = float(os.environ.get('ANSWER_PENALTY_PER_TOKEN', '0.1'))
    min_final_reward = float(os.environ.get('ANSWER_MIN_FINAL_REWARD', '-2.0'))
    # 2. Post-process answers in a batch
    processed_answer_texts = list(answer_texts) # Make a mutable copy
    if enable_postprocessing:
        import os
        use_llm = os.environ.get('USE_LLM_ANSWER_EXTRACTION', 'false').lower() == 'true'
        postprocessor = get_postprocessor()
        # Filter out None answers before batch processing
        answers_to_process_indices = [i for i, at in enumerate(answer_texts) if at is not None]
        answers_to_process = [answer_texts[i] for i in answers_to_process_indices]
        if answers_to_process:
            processed_valid_answers = postprocessor.process_answers_batch(answers_to_process)
            # Place processed answers back into their original positions
            for i, processed_answer in zip(answers_to_process_indices, processed_valid_answers):
                processed_answer_texts[i] = processed_answer
    import os
    use_llm_judge = os.environ.get('USE_LLM_JUDGE', 'false').lower() == 'true'
    # 3. Compute answer scores
    results = []
    for i in range(len(solution_strs)):
        format_correct = format_results[i]
        answer_text = processed_answer_texts[i]
        # Default answerable flag handling: treat None as True
        ans_flag = True
        if answerable_list is not None and i < len(answerable_list) and answerable_list[i] is not None:
            # answerable_list[i] should already be a boolean from naive.py
            # Do NOT use bool() as it converts 'false' string to True!
            ans_flag = answerable_list[i]
            if not isinstance(ans_flag, bool):
                raise ValueError(f"answerable_list[{i}] must be boolean, got {type(ans_flag)}: {ans_flag}")
        if format_correct and answer_text:
            gt = ground_truths[i]
            # Get answer_aliases for this sample
            aliases = answer_aliases_list[i] if answer_aliases_list is not None and i < len(answer_aliases_list) else None
            # Combine ground truth with aliases
            all_answers = []
            if isinstance(gt, list):
                all_answers.extend(gt)
            else:
                all_answers.append(gt)
            if aliases is not None and len(aliases) > 0:
                all_answers.extend(aliases)
            if use_llm_judge:
                postprocessor = get_postprocessor()
                judge_scores = []
                if ans_flag is True:
                    for exp_ans in all_answers:
                        if exp_ans:
                            s = postprocessor.judge_answer_correctness(
                                predicted_answer=answer_text,
                                ground_truth_answer=exp_ans,
                                question=question_texts[i],
                                answerable=ans_flag
                            )
                            judge_scores.append(s)
                accuracy = max(judge_scores) if judge_scores else 0.0
                is_idk = _is_idk_response(answer_text)
                # CRITICAL FIX: Judge may return wrong score, cross-check with IDK heuristic
                if ans_flag is True:
                    if accuracy >= 0.999:
                        # Judge says "correct", but double-check IDK
                        answer_score = 0.0 if is_idk else 1.0
                    elif accuracy <= -0.999:
                        answer_score = -1.0
                    else:
                        # Judge says "IDK" (0) or fallback
                        answer_score = 0.0 if is_idk else -1.0
                else:  # ans_flag is False
                    answer_score = 1.0 if (accuracy >= 1.0 or is_idk) else -1.0
            else:
                metrics = compute_metrics(answer_text, all_answers)
                answer_score = metrics["f1"]
        else:
            # Format invalid or missing answer content → assign -1 as requested
            answer_score = -1.0
        # Store base_reward (original judge result before penalty)
        base_reward = float(answer_score)
        # Apply length penalty if enabled
        if enable_length_penalty and format_correct and answer_text:
            raw_penalty = _calculate_length_penalty(
                answer_text=answer_text,
                max_free_tokens=max_free_tokens,
                penalty_per_token=penalty_per_token,
                max_penalty=float('inf')
            )
            max_applicable_penalty = answer_score - min_final_reward
            length_penalty = min(raw_penalty, max_applicable_penalty)
            answer_score = answer_score - length_penalty
            answer_score = max(answer_score, min_final_reward)
        # Apply repetition penalty
        enable_repetition_penalty = os.environ.get('ENABLE_REPETITION_PENALTY', 'true').lower() == 'true'
        if enable_repetition_penalty and format_correct:
            rep_penalty = _calculate_repetition_penalty(processed_strs[i])
            if rep_penalty > 0:
                answer_score -= rep_penalty
                # Ensure we don't go below min_final_reward
                min_final_reward = float(os.environ.get('ANSWER_MIN_FINAL_REWARD', '-2.0'))
                answer_score = max(answer_score, min_final_reward)
        results.append((format_scores[i], answer_score, base_reward))
    return results