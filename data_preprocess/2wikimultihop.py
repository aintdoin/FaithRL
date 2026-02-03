import os
import sys
# CRITICAL: Must set this before ANY imports that might use torch/CUDA
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import ast
import json
import argparse
import random
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
# Add parent directory to path to import verl modules
# File is in data_preprocess/, so we need to go up one level to reach project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Optional imports - requests for API
try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    REQUESTS_AVAILABLE = False
    print("Warning: requests is not available, API mode will be disabled")
# =========================
# Prompt & dataset helpers
# =========================
def make_prefix_unified(dp: dict, template_type: str) -> str:
    """Unified prompt prefix for both answerable and unanswerable samples"""
    question = dp.get('question', 'no question')
    documents_str = dp.get('documents', '[]')
    try:
        documents_list = ast.literal_eval(documents_str) if isinstance(documents_str, str) else documents_str
    except Exception:
        documents_list = []
    formatted_docs = []
    for doc in documents_list:
        if isinstance(doc, list) and len(doc) == 2:
            title, sentences = doc
            if isinstance(sentences, list):
                text = ' '.join(str(s) for s in sentences)
            else:
                text = str(sentences)
            formatted_docs.append(f"Document '{title}': {text}")
    documents_context = "\n".join(formatted_docs) if formatted_docs else "No references provided."
    user_content = f"""**References:**
{documents_context}
**Question:**
{question}"""
    system_prompt = """You are a helpful assistant. You are given a Question and References.
Your task: answer the Question only using factual information contained in the References. Do not use any external knowledge or your own knowledge.
**CRITICAL - You MUST follow this EXACT format:**
<think>
1. [First reasoning step]
2. [Second reasoning step]
3. [Third reasoning step]
...
</think>
<answer>Your final answer</answer>
**Rules (STRICTLY ENFORCED):**
1. Put reasoning in <think></think> tags
2. Use numbered steps (1., 2., 3., ...) in your <think> section for clear structured reasoning
3. NEVER start with anything other than <think> or <answer>
4. The <answer> tag MUST contain your final answer
Remember: Any response without proper <answer></answer> tags is INCORRECT."""
    if template_type in ['qwen', 'deepseek-r1-distill-qwen', 'deepseek_qwen']:
        prefix = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>"""
    elif template_type in ['llama', 'llama3', 'llama-3']:
        prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Let me solve this step by step.
<think>"""
    else:
        prefix = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>"""
    return prefix, user_content
def gen_from_jsonl(path: str):
    """Load data from JSONL file and convert to dataset format (2WikiMultihop version)"""
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            if 'supporting_facts' in data:
                data['supporting_facts'] = str(data['supporting_facts'])
            if 'evidences' in data:
                try:
                    data['evidences'] = str(data['evidences'])
                except Exception:
                    data['evidences'] = '[]'
            if 'context' in data:
                data['documents'] = str(data['context'])
                del data['context']
            if '_id' in data:
                extra_info = data.get('extra_info', {})
                if not isinstance(extra_info, dict):
                    extra_info = {}
                extra_info['sample_id'] = str(data['_id'])
                data['extra_info'] = extra_info
            yield data
# =========================
# Evidence conversion via API
# =========================
def to_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            return list(ast.literal_eval(s))
        except Exception:
            try:
                return list(json.loads(s))
            except Exception:
                return []
    return []
def clean_evidence_items(items: List[Any]) -> List[List[Any]]:
    result: List[List[Any]] = []
    for item in items:
        if isinstance(item, list) and len(item) == 3:
            result.append(item)
    return result
class TripleSentenceConverter:
    def __init__(self):
        self.api_base = os.environ.get('LLM_JUDGE_API_BASE', '').strip()
        self.model_name = os.environ.get('LLM_JUDGE_MODEL_NAME', '').strip() or 'llm-judge'
        self.api_key = os.environ.get('LLM_JUDGE_API_KEY', '').strip()
        try:
            self.timeout = float(os.environ.get('LLM_JUDGE_TIMEOUT', '60'))
        except Exception:
            self.timeout = 60.0
        try:
            self.max_workers = int(os.environ.get('LLM_JUDGE_MAX_WORKERS', '8'))
        except Exception:
            self.max_workers = 8
        self.requests = requests if REQUESTS_AVAILABLE else None
        self.use_api = bool(self.api_base and self.requests)
    def _build_messages(self, subject: str, relation: str, obj: str) -> list:
        system_content = (
            "You are an expert at converting knowledge triples into clear, natural English sentences.\n\n"
            "Task Instructions:\n"
            "1. Transform the triple into ONE grammatically correct sentence\n"
            "2. Maintain the semantic relationship between the subject and object\n"
            "3. Use appropriate phrasing based on the relation type\n"
            "4. Return ONLY the resulting sentence, nothing else\n\n"
            "Example:\n"
            "Triple: ['Stuart Rosenberg', 'director', 'Move (1970 film)']\n"
            "Output: Stuart Rosenberg is the director of Move (1970 film).\n\n"
            "Triple: ['Jean-Daniel Pollet', 'country of citizenship', 'French']\n"
            "Output: Jean-Daniel Pollet's country of citizenship is France."
        )
        user_content = (
            "Convert the following knowledge triple into a single, natural English sentence:\n"
            f"['{subject}', '{relation}', '{obj}']"
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
    def _call_chat(self, messages: list) -> str:
        assert self.requests is not None
        base = self.api_base.rstrip('/')
        url = base + '/v1/chat/completions'
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        payload = {
            'model': self.model_name,
            'messages': messages,
            'temperature': 0.0,
            'max_tokens': 80,
            'stream': False,
        }
        resp = self.requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        return text.strip() if text else ''
    def convert_triple(self, triple: List[Any]) -> str:
        subject, relation, obj = (str(triple[0]), str(triple[1]), str(triple[2]))
        if self.use_api:
            try:
                messages = self._build_messages(subject, relation, obj)
                out = self._call_chat(messages)
                if out:
                    return out
            except Exception:
                pass
        return f"{subject} {relation} {obj}."
    def convert_triples(self, triples: List[List[Any]]) -> List[str]:
        if not triples:
            return []
        # Simple sequential for determinism and avoiding too many threads in data preprocess
        return [self.convert_triple(t) for t in triples]
def convert_evidences_to_sentences(evidences_cell: Any, converter: TripleSentenceConverter) -> List[str]:
    items = to_list(evidences_cell)
    triples = clean_evidence_items(items)
    sentences = converter.convert_triples(triples)
    return sentences
# =========================
# Best-of-N filtering utils
# =========================
def _has_valid_format(text: str) -> bool:
    try:
        a_s = text.count('<answer>')
        a_e = text.count('</answer>')
        if a_s != 1 or a_e != 1:
            return False
        ps = text.find('<answer>')
        pe = text.find('</answer>')
        if ps == -1 or pe == -1 or ps >= pe:
            return False
        content = text[ps + len('<answer>'):pe].strip()
        if len(content) == 0:
            return False
        return True
    except Exception:
        return False
def _is_idk_answer(text: str) -> bool:
    if not text:
        return False
    text_lower = text.strip().lower()
    idk_markers = [
        "i don't know", "i dont know", "i do not know",
        "i'm not sure", "i am not sure", "not sure",
        "cannot answer", "can't answer", "unable to answer",
        "cannot determine", "can't determine", "unable to determine",
        "insufficient information", "not enough information",
        "no sufficient information", "lack of information",
        "unknown", "unclear", "uncertain",
    ]
    return any(marker in text_lower for marker in idk_markers)
def call_api_for_candidates(prompt: str, api_base: str, model_name: str, api_key: str,
                            n: int, temperature: float, top_p: float, top_k: int, max_tokens: int) -> list:

    base = api_base.rstrip('/')
    chat_url = base + '/v1/chat/completions'
    comp_url = base + '/v1/completions'
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    chat_payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
        'n': n,
        'stream': False,
    }
    resp = requests.post(chat_url, json=chat_payload, headers=headers, timeout=120)
    if resp.status_code == 200:
        data = resp.json()
        candidates = []
        try:
            for choice in data.get('choices', []):
                text = choice.get('message', {}).get('content', '')
                if text and text.strip():
                    candidates.append(text.strip())
        except Exception:
            pass
        if candidates:
            return candidates
    comp_payload = {
        'model': model_name,
        'prompt': prompt,
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
        'n': n,
        'stream': False,
    }
    resp2 = requests.post(comp_url, json=comp_payload, headers=headers, timeout=120)
    data2 = resp2.json()
    candidates = []
    for choice in data2.get('choices', []):
        text = choice.get('text', '').strip()
        if text:
            candidates.append(text)
    return candidates if candidates else []
def evaluate_sample_best_of_n(sample_dict: dict, prompt: str, args, llm, sampling_params, postprocessor):
    import re
    extra_info = sample_dict.get('extra_info', {})
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except Exception:
            extra_info = {}
    question = sample_dict.get('question', '')
    ground_truth = sample_dict.get('answer', '')
    answer_aliases = extra_info.get('answer_aliases', [])
    if isinstance(answer_aliases, np.ndarray):
        answer_aliases = answer_aliases.tolist()
    elif answer_aliases is None:
        answer_aliases = []
    if args.use_api:
        candidates = call_api_for_candidates(
            prompt, args.api_base, args.model_name, args.api_key,
            args.n_candidates, args.temperature, args.top_p, args.top_k, args.max_tokens
        )
        has_correct_answer = False
        best_reward = -1
        for candidate_text in candidates:
            if not _has_valid_format(candidate_text):
                reward = -1
            else:
                match = re.search(r'<answer>\s*(.*?)\s*</answer>', candidate_text, re.DOTALL | re.IGNORECASE)
                extracted_answer = match.group(1).strip() if match else candidate_text.strip()
                is_idk = _is_idk_answer(extracted_answer)
                reward_scores = []
                all_answers = [ground_truth]
                if answer_aliases and len(answer_aliases) > 0:
                    all_answers.extend(answer_aliases)
                for ans in all_answers:
                    if ans:
                        try:
                            score = postprocessor.judge_answer_correctness(
                                predicted_answer=candidate_text,
                                ground_truth_answer=ans,
                                question=question,
                                answerable=False
                            )
                            reward_scores.append(score)
                        except Exception:
                            continue
                reward = max(reward_scores) if reward_scores else 0
                if reward >= 0.999 and not is_idk:
                    has_correct_answer = True
            if reward > best_reward:
                best_reward = reward
        return not has_correct_answer, best_reward
    else:
        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0]
        has_correct_answer = False
        best_reward = -1
        for candidate_output in output.outputs:
            generated_text = candidate_output.text
            if not _has_valid_format(generated_text):
                reward = -1
            else:
                match = re.search(r'<answer>\s*(.*?)\s*</answer>', generated_text, re.DOTALL | re.IGNORECASE)
                extracted_answer = match.group(1).strip() if match else generated_text.strip()
                is_idk = _is_idk_answer(extracted_answer)
                reward_scores = []
                all_answers = [ground_truth]
                if answer_aliases and len(answer_aliases) > 0:
                    all_answers.extend(answer_aliases)
                for ans in all_answers:
                    if ans:
                        try:
                            score = postprocessor.judge_answer_correctness(
                                predicted_answer=generated_text,
                                ground_truth_answer=ans,
                                question=question,
                                answerable=False
                            )
                            reward_scores.append(score)
                        except Exception:
                            continue
                reward = max(reward_scores) if reward_scores else 0
                if reward >= 0.999 and not is_idk:
                    has_correct_answer = True
            if reward > best_reward:
                best_reward = reward
        return not has_correct_answer, best_reward
# =========================
# Sample builders
# =========================
def create_answerable_samples(dataset: Dataset, num_samples: int, template_type: str,
                              converter: TripleSentenceConverter) -> List[dict]:
    answerable_dataset = dataset.select(range(num_samples))
    answerable_samples: List[dict] = []
    for i in range(len(answerable_dataset)):
        sample = answerable_dataset[i]
        evid_sentences = convert_evidences_to_sentences(sample.get('evidences', '[]'), converter)
        answerable_sample = {
            'question': sample.get('question', ''),
            'documents': sample.get('documents', '[]'),
            'answer': sample['answer'],
            'data_source': sample.get('data_source', '2wikimultihop'),
            'evidences': evid_sentences,  
            'extra_info': sample.get('extra_info', {}),
        }
        if isinstance(answerable_sample['extra_info'], dict):
            answerable_sample['extra_info']['answerable'] = True
        else:
            answerable_sample['extra_info'] = {'answerable': True}
        answerable_samples.append(answerable_sample)
    return answerable_samples
def create_unanswerable_samples_with_filter(dataset: Dataset, start_idx: int, num_samples: int,
                                            template_type: str, args, llm, sampling_params,
                                            postprocessor, converter: TripleSentenceConverter) -> List[dict]:
    kept_samples: List[dict] = []
    removed_count = 0
    processed_count = 0
    pbar = tqdm(total=num_samples, desc=" ")
    idx = start_idx
    while len(kept_samples) < num_samples and idx < len(dataset) - 1:
        question_sample = dataset[idx]
        question = question_sample.get('question', '')
        answer = question_sample['answer']
        try:
            original_documents = ast.literal_eval(question_sample.get('documents', '[]'))
        except Exception:
            original_documents = []
        supporting_facts = question_sample.get('supporting_facts', None)
        if supporting_facts is None:
            extra_info = question_sample.get('extra_info', {})
            if isinstance(extra_info, str):
                try:
                    extra_info = ast.literal_eval(extra_info)
                except Exception:
                    extra_info = {}
            supporting_facts = extra_info.get('supporting_facts', [])
        if isinstance(supporting_facts, str):
            try:
                supporting_facts = ast.literal_eval(supporting_facts)
            except Exception:
                supporting_facts = []
        supporting_doc_titles: List[str] = []
        if isinstance(supporting_facts, list):
            for fact in supporting_facts:
                if isinstance(fact, list) and len(fact) >= 1:
                    title = fact[0]
                    if title not in supporting_doc_titles:
                        supporting_doc_titles.append(title)
        if supporting_doc_titles and len(supporting_doc_titles) > 1 and original_documents:
            removal_candidates = supporting_doc_titles[1:]
            if len(supporting_doc_titles) >= 4:
                num_to_remove = min(2, len(removal_candidates))
            else:
                num_to_remove = 1
            docs_to_remove = random.sample(removal_candidates, num_to_remove)
            modified_documents = original_documents.copy()
            for doc_title in docs_to_remove:
                modified_documents = [doc for doc in modified_documents 
                                      if not (isinstance(doc, list) and len(doc) >= 2 and doc[0] == doc_title)]
        else:
            modified_documents = original_documents
        evid_sentences = convert_evidences_to_sentences(question_sample.get('evidences', '[]'), converter)
        unanswerable_sample = {
            'question': question,
            'documents': str(modified_documents),
            'answer': answer,
            'data_source': question_sample.get('data_source', '2wikimultihop'),
            'evidences': evid_sentences,
            'extra_info': question_sample.get('extra_info', {}),
        }
        if isinstance(unanswerable_sample['extra_info'], dict):
            unanswerable_sample['extra_info']['answerable'] = False
        else:
            unanswerable_sample['extra_info'] = {'answerable': False}
        prompt, _ = make_prefix_unified(unanswerable_sample, template_type)
        is_truly_unanswerable, best_reward = evaluate_sample_best_of_n(
            unanswerable_sample, prompt, args, llm, sampling_params, postprocessor
        )
        processed_count += 1


        idx += 1
        if idx >= len(dataset) - 1:
            break
    pbar.close()
    return kept_samples
# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description='2wikimultihop conversion + evidences sentence conversion + Best-of-N filtering (save answerable/unanswerable separately)')
    parser.add_argument('--type', type=str, default='train', help='train or test')
    parser.add_argument('--template_type', type=str, default='deepseek-r1-distill-qwen')
    parser.add_argument('--size', type=int, required=True, help='Total target sample size (split equally between answerable and unanswerable)')
    parser.add_argument('--data-path', type=str, default=None, help='Input JSONL file path')
    parser.add_argument('--model-path', type=str, default='', help='Local model path (vLLM mode)')
    parser.add_argument('--use-api', action='store_true', help='Use API mode instead of local vLLM (for Best-of-N)')
    parser.add_argument('--api-base', type=str, default='http://localhost:8000', help='API base URL')
    parser.add_argument('--api-key', type=str, default='', help='API key (optional)')
    parser.add_argument('--model-name', type=str, default='', help='API model name')
    parser.add_argument('--n-candidates', type=int, default=32, help='Number of candidate answers generated per sample')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p sampling parameter')
    parser.add_argument('--top-k', type=int, default=100, help='Top-k sampling parameter')
    parser.add_argument('--max-tokens', type=int, default=2048, help='Maximum number of generated tokens')
    parser.add_argument('--max-model-len', type=int, default=24500, help='vLLM maximum model length')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='vLLM tensor parallel size')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("2WIKIMULTIHOP CONVERSION + EVIDENCES SENTENCE CONVERSION + BEST-OF-N FILTERING")
    print("="*80)
    print(f"Total target samples: {args.size}")
    answerable_size = args.size // 2
    unanswerable_size = args.size - answerable_size
    print(f"  - Answerable(True): {answerable_size}")
    print(f"  - Unanswerable(False, need filtering): {unanswerable_size}")
    print(f"Filter mode: {'API' if args.use_api else 'Local vLLM'}")
    
    if args.use_api:
        print(f"API Base: {args.api_base}")
        print(f"Model: {args.model_name}")
    else:
        print(f"Model path: {args.model_path}")
    
    print(f"Best-of-N: {args.n_candidates}")
    print("="*80)
    
    if args.data_path:
        data_path = args.data_path
    elif args.type == 'train':
        data_path = './2wikimultihop/data/train.jsonl'
    else:
        data_path = './2wikimultihop/data/dev.jsonl'
    
    print(f"\n📂 Loading data from {data_path}...")
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': data_path})
    print(f"   ✓ Raw dataset length: {len(raw_dataset)}")
    
    total_needed = answerable_size + unanswerable_size * 10
    dataset = raw_dataset.shuffle(seed=42).select(range(min(total_needed, len(raw_dataset))))
    print(f"   ✓ Selected {len(dataset)} samples for processing")
    
    llm = None
    sampling_params = None
    
    if args.use_api:
        print(f"\n🌐 Testing API connection (for Best-of-N)...")
        if not REQUESTS_AVAILABLE:
            print("Error: requests library is not available!")
            return
        if not args.model_name:
            print("Error: --model-name parameter is required for API mode!")
            return
        
        base = args.api_base.rstrip('/')
        models_url = base + '/v1/models'
        
        try:
            resp = requests.get(models_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                available_models = [m.get('id', 'unknown') for m in data.get('data', [])]
                print(f"  ✓ API accessible")
                print(f"  Available models: {available_models}")
                if args.model_name not in available_models and available_models:
                    print(f"  ⚠️ Warning: '{args.model_name}' not in available models list")
            else:
                print(f"  ⚠️ Warning: Cannot access models endpoint (Status: {resp.status_code})")
        except Exception as e:
            print(f"  ⚠️ Warning: Cannot connect to API: {e}")
            return
    else:
        print(f"\n🔧 Loading model from {args.model_path} for Best-of-N...")
        if not args.model_path:
            print("Error: --model-path parameter is required for local mode!")
            return
        
        try:
            from vllm import LLM, SamplingParams  # type: ignore
            print("   ✓ vLLM module imported successfully")
        except Exception:
            print("Error: vLLM is not available! Please use --use-api to switch to API mode.")
            return
        
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
            max_tokens=args.max_tokens,
            n=args.n_candidates,
        )
        print("   ✓ Model loaded successfully")
    
    print("\n🔍 Initializing answer postprocessor...")
    from verl.utils.reward_score.answer_postprocessor import get_postprocessor  # type: ignore
    postprocessor = get_postprocessor()
    print("   ✓ Postprocessor initialized successfully")
    
    converter = TripleSentenceConverter()
    if converter.use_api:
        print("   ✓ Evidence conversion will use Chat Completions API")
    else:
        print("   ⚠️ Evidence conversion will use fallback template (API not configured or requests unavailable)")
    
    answerable_samples = create_answerable_samples(dataset, answerable_size, args.template_type, converter)
    
    def build_row_with_prompt(example: dict) -> dict:
        _, question_prefixed = make_prefix_unified(example, template_type=args.template_type)
        return {
            "prompt": question_prefixed,
            "question": example['question'],
            "answer": example['answer'],
            "data_source": example['data_source'],
            "extra_info": example['extra_info'],
            "documents": example['documents'],
            "evidences": example['evidences'],
        }
    
    ans_true_ds = Dataset.from_list(answerable_samples)
    print("\nGenerating prompts for answerable=True samples...")
    ans_true_ds = ans_true_ds.map(lambda ex, idx: build_row_with_prompt(ex), with_indices=True)
    
    output_dir = f'data/2wikimultihop/{args.template_type}'
    os.makedirs(os.path.expanduser(output_dir), exist_ok=True)
    
    ans_true_path = os.path.join(output_dir, '2wikimultihop_ans_true.parquet')
    ans_true_ds.to_parquet(ans_true_path)
    print(f"💾 Saved answerable=True to {ans_true_path} (samples: {len(ans_true_ds)})")
    
    unanswerable_samples = create_unanswerable_samples_with_filter(
        dataset, answerable_size, unanswerable_size, args.template_type,
        args, llm, sampling_params, postprocessor, converter
    )
    
    ans_false_ds = Dataset.from_list(unanswerable_samples)
    print("\nGenerating prompts for answerable=False samples...")
    ans_false_ds = ans_false_ds.map(lambda ex, idx: build_row_with_prompt(ex), with_indices=True)
    ans_false_ds = ans_false_ds.shuffle(seed=42)
    
    ans_false_path = os.path.join(output_dir, '2wikimultihop_ans_false.parquet')
    ans_false_ds.to_parquet(ans_false_path)
    print(f"💾 Saved answerable=False to {ans_false_path} (samples: {len(ans_false_ds)})")
    
    print("\nVerifying saved results...")
    df_true = pd.read_parquet(ans_true_path)
    df_false = pd.read_parquet(ans_false_path)
    
    n_true = sum(
        1 for _, row in df_true.iterrows()
        if isinstance(row.get('extra_info'), (dict, str)) and
        (json.loads(row['extra_info']) if isinstance(row['extra_info'], str) else row['extra_info']).get('answerable') is True
    )
    
    n_false = sum(
        1 for _, row in df_false.iterrows()
        if isinstance(row.get('extra_info'), (dict, str)) and
        (json.loads(row['extra_info']) if isinstance(row['extra_info'], str) else row['extra_info']).get('answerable') is False
    )
    
    print(f"   ✓ ans_true answerable=True: {n_true}/{len(df_true)}")
    print(f"   ✓ ans_false answerable=False: {n_false}/{len(df_false)}")
    
    if llm is not None:
        try:
            llm.shutdown()
        except Exception:
            pass
    
    print("\n✅ Completed!\n")

if __name__ == '__main__':
    main()