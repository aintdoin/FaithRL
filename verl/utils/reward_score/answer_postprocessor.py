import os
import re
import concurrent.futures
from typing import Optional
import threading
import hashlib
from collections import OrderedDict
# Global singleton
_global_postprocessor = None
class AnswerPostProcessor:
    """
    Minimal answer post-processor focused on LLM-as-a-Judge.
    """
    def __init__(self):
        """
        Initialize with LLM Judge API configuration only.
        """
        # LLM Judge API configuration
        self.judge_api_base = os.environ.get('LLM_JUDGE_API_BASE', '').strip()
        self.judge_model_name = os.environ.get('LLM_JUDGE_MODEL_NAME', '').strip()
        self.judge_api_key = os.environ.get('LLM_JUDGE_API_KEY', '').strip()
        self.use_judge_api = bool(self.judge_api_base)
        # Check requests library availability
        try:
            import requests
            self.requests = requests
        except ImportError:
            self.requests = None
            if self.use_judge_api:
                print("⚠️  WARNING: requests library not available, LLM Judge will be disabled")
                self.use_judge_api = False
        # Concurrency settings
        self.max_workers = int(os.environ.get('LLM_JUDGE_MAX_WORKERS', '8'))
        # Increase timeout to avoid empty returns under load
        self.request_timeout = float(os.environ.get('LLM_JUDGE_TIMEOUT', '60'))
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) if self.use_judge_api else None
        # Debug logging
        print(f"[AnswerPostProcessor] Judge cfg -> base={bool(self.judge_api_base)}, model={'set' if self.judge_model_name else 'unset'}, workers={self.max_workers}, timeout={self.request_timeout}s")
        # ---- FLOPs accounting (server-side LLM judge) ----
        self._flops_lock = threading.Lock()
        self._judge_server_flops_total = 0.0  # raw FLOPs
        self._judge_flops_counter = None
        # ---- Prefix/KV cache accounting (best-effort) ----
        # When vLLM prefix caching is enabled, repeated prompt prefixes may reuse KV cache.
        # To keep FLOPs accounting closer to real compute, we optionally discount cached
        # prefix tokens after the first time a given prefix is seen (per-process).
        self._prefix_cache_lock = threading.Lock()
        self._prefix_token_cache: "OrderedDict[str, int]" = OrderedDict()
        self._prefix_cache_max_entries = int(os.environ.get("LLM_JUDGE_PREFIX_CACHE_MAX_ENTRIES", "2048"))
    def consume_judge_server_flops(self) -> float:
        """
        Consume and reset accumulated judge-server FLOPs since last call.
        """
        with self._flops_lock:
            v = float(self._judge_server_flops_total)
            self._judge_server_flops_total = 0.0
            return v
    def _maybe_init_judge_flops_counter(self):
        if self._judge_flops_counter is not None:
            return
        try:
            from transformers import AutoConfig
            from verl.utils.flops_counter import FlopsCounter
            if not self.judge_model_name:
                return
            cfg = AutoConfig.from_pretrained(self.judge_model_name, trust_remote_code=True)
            self._judge_flops_counter = FlopsCounter(cfg)
        except Exception:
            self._judge_flops_counter = None
    def _maybe_init_judge_tokenizer(self):
        if hasattr(self, "_judge_tokenizer"):
            return
        self._judge_tokenizer = None
        try:
            from transformers import AutoTokenizer
            if not self.judge_model_name:
                return
            self._judge_tokenizer = AutoTokenizer.from_pretrained(self.judge_model_name, trust_remote_code=True)
        except Exception:
            self._judge_tokenizer = None
    def process_answer(self, answer_text: str, expected_answer: str = None, question: str = None) -> str:
        """
        Apply only basic normalization (no LLM extraction).
        Args:
            answer_text: Raw answer from model
            expected_answer: Not used (kept for compatibility)
            question: Not used (kept for compatibility)
        Returns:
            Normalized answer string (basic normalization only)
        """
        if not answer_text:
            return ""
        # Basic normalization only
        answer = answer_text.strip()
        # Remove common prefixes
        prefixes = [
            "the answer is:",
            "the final answer is:",
            "answer:",
            "final answer:",
        ]
        answer_lower = answer.lower()
        for prefix in prefixes:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        # Remove surrounding quotes
        if (answer.startswith('"') and answer.endswith('"')) or \
           (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1].strip()
        return answer
    def process_answers_batch(self, answer_texts: list) -> list:
        """
        Apply basic normalization to a batch of answers.
        Args:
            answer_texts: List of raw answer strings from model
        Returns:
            List of normalized answer strings
        """
        return [self.process_answer(answer_text) for answer_text in answer_texts]
    def _rule_based_match(self, predicted: str, ground_truth: str) -> bool:
        """
        Only case-insensitive exact match.
        """
        if predicted is None or ground_truth is None:
            return False
        return str(predicted).strip().lower() == str(ground_truth).strip().lower()
    def _estimate_text_tokens(self, text: str) -> int:
        if not isinstance(text, str) or not text:
            return 0
        try:
            self._maybe_init_judge_tokenizer()
            tok = getattr(self, "_judge_tokenizer", None)
            if tok is None:
                return 0
            return int(len(tok.encode(text, add_special_tokens=False)))
        except Exception:
            return 0
    def _prefix_cache_key(self, prefix: str) -> str:
        try:
            b = prefix.encode("utf-8", errors="ignore")
        except Exception:
            b = bytes(str(prefix), "utf-8", errors="ignore")
        return hashlib.sha1(b).hexdigest()
    def _call_judge_api(
        self,
        prompt: str,
        *,
        cache_prefix: Optional[str] = None,
        cache_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 5,
    ) -> str:
        """
        Internal helper to call the LLM Judge API.
        Returns the raw response text.
        """
        # Prefer chat-completions path for vLLM OpenAI server; fall back to completions
        base = self.judge_api_base.rstrip('/')
        chat_url = base + '/v1/chat/completions'
        comp_url = base + '/v1/completions'
        model_name = self.judge_model_name or 'llm-judge'
        headers = {'Content-Type': 'application/json'}
        if self.judge_api_key:
            headers['Authorization'] = f'Bearer {self.judge_api_key}'
        # Try chat endpoint first
        chat_payload = {
            'model': model_name,
            'messages': [
                {'role': 'system', 'content': system_prompt or 'You are a strict answer evaluator. Output only a single digit.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.0,
            'max_tokens': int(max_tokens),
            'stream': False,
        }
        resp = self.requests.post(chat_url, json=chat_payload, headers=headers, timeout=self.request_timeout)
        if resp.status_code == 200:
            data = resp.json()
            text = ''
            try:
                text = data['choices'][0]['message']['content']
            except Exception:
                text = ''
            # FLOPs accounting (judge-server forward) with best-effort prefix-cache adjustment.
            # We prefer server usage fields, but when cache_prefix is provided we estimate
            # prompt tokens ourselves and discount cached prefix tokens after first use.
            try:
                usage = data.get('usage', {}) if isinstance(data, dict) else {}
                pt = int((usage.get('prompt_tokens', 0) or 0)) if isinstance(usage, dict) else 0
                ct = int((usage.get('completion_tokens', 0) or 0)) if isinstance(usage, dict) else 0
                use_prefix_accounting = os.environ.get('LLM_JUDGE_PREFIX_CACHE_ACCOUNTING', 'true').lower() == 'true'
                # Prompt tokens (effective)
                if use_prefix_accounting and isinstance(cache_prefix, str) and cache_prefix and isinstance(prompt, str) and prompt.startswith(cache_prefix):
                    suffix = prompt[len(cache_prefix):]
                    prefix_tokens = self._estimate_text_tokens(cache_prefix)
                    suffix_tokens = self._estimate_text_tokens(suffix)
                    k = str(cache_key) if cache_key else self._prefix_cache_key(cache_prefix)
                    with self._prefix_cache_lock:
                        seen = k in self._prefix_token_cache
                        if not seen:
                            self._prefix_token_cache[k] = int(prefix_tokens)
                            if len(self._prefix_token_cache) > self._prefix_cache_max_entries:
                                self._prefix_token_cache.popitem(last=False)
                        else:
                            self._prefix_token_cache.move_to_end(k)
                    prompt_tokens_effective = int(suffix_tokens) if seen else int(prefix_tokens + suffix_tokens)
                else:
                    prompt_tokens_effective = int(pt) if pt > 0 else int(self._estimate_text_tokens(prompt))
                # Completion tokens (always computed)
                completion_tokens = int(ct) if ct > 0 else int(self._estimate_text_tokens(text or ''))
                total_tokens_effective = int(prompt_tokens_effective + completion_tokens)
                if total_tokens_effective > 0:
                    self._maybe_init_judge_flops_counter()
                    if self._judge_flops_counter is not None:
                        flops = float(self._judge_flops_counter.estimate_total_flops_forward_linear(total_tokens_effective))
                        with self._flops_lock:
                            self._judge_server_flops_total += flops
            except Exception:
                pass
            if text and text.strip():
                return text.strip()
        # Fallback to completions endpoint
        comp_payload = {
            'model': model_name,
            'prompt': prompt,
            'temperature': 0.0,
            'max_tokens': 8,
            'stream': False,
            'stop': ['\n']
        }
        resp2 = self.requests.post(comp_url, json=comp_payload, headers=headers, timeout=self.request_timeout)
        if resp2.status_code != 200:
            raise Exception(f"API error: {resp2.status_code}, {resp2.text[:200]}")
        data2 = resp2.json()
        text2 = data2.get('choices', [{}])[0].get('text', '').strip()
        # FLOPs accounting (judge-server forward) with best-effort prefix-cache adjustment
        try:
            usage = data2.get('usage', {}) if isinstance(data2, dict) else {}
            pt = int((usage.get('prompt_tokens', 0) or 0)) if isinstance(usage, dict) else 0
            ct = int((usage.get('completion_tokens', 0) or 0)) if isinstance(usage, dict) else 0
            use_prefix_accounting = os.environ.get('LLM_JUDGE_PREFIX_CACHE_ACCOUNTING', 'true').lower() == 'true'
            if use_prefix_accounting and isinstance(cache_prefix, str) and cache_prefix and isinstance(prompt, str) and prompt.startswith(cache_prefix):
                suffix = prompt[len(cache_prefix):]
                prefix_tokens = self._estimate_text_tokens(cache_prefix)
                suffix_tokens = self._estimate_text_tokens(suffix)
                k = str(cache_key) if cache_key else self._prefix_cache_key(cache_prefix)
                with self._prefix_cache_lock:
                    seen = k in self._prefix_token_cache
                    if not seen:
                        self._prefix_token_cache[k] = int(prefix_tokens)
                        if len(self._prefix_token_cache) > self._prefix_cache_max_entries:
                            self._prefix_token_cache.popitem(last=False)
                    else:
                        self._prefix_token_cache.move_to_end(k)
                prompt_tokens_effective = int(suffix_tokens) if seen else int(prefix_tokens + suffix_tokens)
            else:
                prompt_tokens_effective = int(pt) if pt > 0 else int(self._estimate_text_tokens(prompt))
            completion_tokens = int(ct) if ct > 0 else int(self._estimate_text_tokens(text2 or ''))
            total_tokens_effective = int(prompt_tokens_effective + completion_tokens)
            if total_tokens_effective > 0:
                self._maybe_init_judge_flops_counter()
                if self._judge_flops_counter is not None:
                    flops = float(self._judge_flops_counter.estimate_total_flops_forward_linear(total_tokens_effective))
                    with self._flops_lock:
                        self._judge_server_flops_total += flops
        except Exception:
            pass
        # Ensure non-empty return to avoid downstream ambiguity (default to '-1')
        return text2 if text2 else '-1'
    def _extract_final_answer(self, text: str) -> str:
        """
        Extract final answer from <answer>...</answer> tags.
        If tags are not present, return the original text.
        """
        if not text:
            return ""
        # Extract text between <answer> and </answer> tags
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # If no tags found, return original text
        return text.strip()
    def _clean_question(self, question: str) -> str:
        """
        No preprocessing needed for question - return as-is.
        """
        return question if question else ""
    def _is_idk_response(self, text: str) -> bool:
        """
        Strict IDK exact match (case-insensitive only).
        """
        if not text:
            return False
        text_lower = text.strip().lower()
        idk_markers = [
            "i don't know", "i dont know",
            "insufficient information", 
            "unknown"
        ]
        return any(text_lower == marker for marker in idk_markers)
    def judge_answer_correctness(self, predicted_answer: str, ground_truth_answer: str, question: str = None, answerable: Optional[bool] = None) -> int:
        """
        Simplified three-step judgment:
        1) Case-insensitive exact match with GT -> return 1;
        2) Case-insensitive exact match with IDK list -> return 0 if answerable=True; return 1 if answerable=False; return 0 if answerable=None;
        3) Others are handled by LLM-as-a-Judge (using the provided prompt uniformly, output -1 or 1).
        """
        # Preprocess inputs - extract final answer and clean question
        predicted_answer = self._extract_final_answer(predicted_answer) if predicted_answer else ""
        question = self._clean_question(question) if question else None
        # STAGE 1: Rule-based matching (high-confidence correct cases)
        # This stage can ONLY return True (confident match) or False (uncertain, needs LLM)
        if self._rule_based_match(predicted_answer, ground_truth_answer):
            # High-confidence match found via rule-based methods
            # For answerable questions, this is definitely correct (score=1)
            # For unanswerable questions, matching GT also means correct (score=1)
            # Note: We don't need to check answerable here because a match is a match
            return 1
        # STAGE 2: Rule-based IDK detection (NEW - critical for correctness!)
        # This prevents LLM parsing errors on "I don't know" responses
        if self._is_idk_response(predicted_answer):
            if answerable is True:
                return 0
            if answerable is False:
                return 1
            return 0
        # STAGE 3: LLM Judge (uncertain cases only)
        # Check if we have API configured
        if not self.use_judge_api:
            return -1
        # To maximize vLLM prefix/KV-cache reuse across multiple rollouts (same question/GT, different predictions),
        # we split the prompt into [prefix (static + question/GT)] + [suffix (prediction)].
        prefix = f"""Input: Assume you are a human expert in grading predictions given by a model. You are given a question and a model prediction. Judge if the prediction matches the ground truth answer by following these steps:
1: Take it as granted that the Ground Truth is always correct.
2: If the Prediction exactly matches the Ground Truth, “score" is 1.
3: If the Ground Truth is a number, “score" is 1 if and only if the Prediction gives a number that almost exactly matches the ground truth.
4: If the Prediction is self-contradictory, “score" must be -1.
5: If the prediction is not answering the question, “score" must be -1.
6: If the prediction is a concise and correct summary of the ground truth, “score" is 1.
7: If ground truth contains a set of items, prediction must contain exactly same items for the score to be 1.
8: Otherwise, “score" is -1.
Output ONLY one digit: -1 or 1. No text, no explanation.
You should make the judgment based on provided examples.
Examples: 
Question: When did the director of film Lord Richard In The Pantry die?
Ground Truth: 7 January 1984
Prediction: January 7, 1984
Output: 1
Question: Who is older, Charles Badham or Médéric De Vasselot De Régné?
Ground Truth: Charles Badham
Prediction: Médéric De Vasselot De Régné
Output: -1
        Question: {question}
        Ground Truth: {ground_truth_answer}
        Prediction: """
        suffix = f"""{predicted_answer}
        Output: """
        prompt = prefix + suffix
        # Call judge API (using thread pool for async execution)
        try:
            _cache_key = "final_correctness_prefix:" + hashlib.sha1(prefix.encode("utf-8", errors="ignore")).hexdigest()
            result = self._call_judge_api(prompt, cache_prefix=prefix, cache_key=_cache_key)
            # Parse result - should be single digit: -1 or 1
            result_clean = result.strip()
            # ROBUST parsing: handle various output formats
            # 1. Try exact match first (best case)
            if result_clean in ['-1', '1']:
                score = int(result_clean)
            else:
                # 2. Try to find digit at the START (avoid parsing list numbers like "1. ")
                import re
                # Match digit at start of string, possibly with whitespace
                match = re.match(r'^\s*(-?\d+)', result_clean)
                if match:
                    score = int(match.group(1))
                else:
                    # 3. Last resort: search anywhere (but this is risky)
                    match = re.search(r'-?\d+', result_clean)
                    if match:
                        score = int(match.group())
                    else:
                        raise ValueError(f"No number found in: {result}")
            # Validate score range
            if score == -1:
                return -1
            elif score == 1:
                return 1
            else:
                # Out of range, use fallback
                raise ValueError(f"Score {score} not in {{-1, 1}}")
        except Exception as e:
            return -1
    def judge_prompt_batch(self, prompts: list[dict]) -> list[str]:
        """
        Submit a batch of raw judge prompts concurrently.

        Each item should contain:
            - prompt: full prompt string
            - cache_prefix: optional shared prefix for cache accounting
            - cache_key: optional stable cache key
            - system_prompt: optional system prompt
            - max_tokens: optional max token cap
        """
        if not prompts:
            return []
        if not self.use_judge_api or not self.executor:
            results = []
            for item in prompts:
                try:
                    results.append(self._call_judge_api(
                        item['prompt'],
                        cache_prefix=item.get('cache_prefix'),
                        cache_key=item.get('cache_key'),
                        system_prompt=item.get('system_prompt'),
                        max_tokens=int(item.get('max_tokens', 5)),
                    ))
                except Exception:
                    results.append('')
            return results
        futures = {}
        for idx, item in enumerate(prompts):
            futures[self.executor.submit(
                self._call_judge_api,
                item['prompt'],
                cache_prefix=item.get('cache_prefix'),
                cache_key=item.get('cache_key'),
                system_prompt=item.get('system_prompt'),
                max_tokens=int(item.get('max_tokens', 5)),
            )] = idx
        results = [''] * len(prompts)
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result(timeout=self.request_timeout + 10)
            except Exception:
                results[idx] = ''
        return results
    def judge_batch(self, items: list) -> list:
        """
        Batch judge multiple answers concurrently using thread pool.
        Args:
            items: List of dicts, each containing:
                - predicted_answer: str
                - ground_truth_answer: str
                - question: str (optional)
                - answerable: bool (optional)
        Returns:
            List of scores (int) in the same order as input items
        """
        if not self.use_judge_api or not self.executor:
            # Fallback to sequential processing
            return [
                self.judge_answer_correctness(
                    item['predicted_answer'],
                    item['ground_truth_answer'],
                    item.get('question'),
                    item.get('answerable')
                )
                for item in items
            ]
        # Submit all tasks to thread pool
        futures = []
        for item in items:
            future = self.executor.submit(
                self.judge_answer_correctness,
                item['predicted_answer'],
                item['ground_truth_answer'],
                item.get('question'),
                item.get('answerable')
            )
            futures.append(future)
        # Collect results in order
        results = []
        for future in futures:
            try:
                result = future.result(timeout=self.request_timeout + 10)
                results.append(result)
            except Exception as e:
                results.append(0)  # Default neutral score on failure
        return results
    def shutdown(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=False)
def get_postprocessor() -> AnswerPostProcessor:
    """Get or create the global post-processor instance"""
    global _global_postprocessor
    if _global_postprocessor is None:
        _global_postprocessor = AnswerPostProcessor()
    return _global_postprocessor
