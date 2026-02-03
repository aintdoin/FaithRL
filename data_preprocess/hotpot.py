import os
import sys
# CRITICAL: Must set this before ANY imports that might use torch/CUDA
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
from datasets import Dataset
import ast
import random
# Add parent directory to path to import verl modules
# File is in data_preprocess/, so we need to go up one level to reach project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Optional imports - requests for API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests is not available, cannot use API mode")

def make_prefix_unified(dp, template_type):
    """Unified prompt prefix for both answerable and unanswerable samples"""
    question = dp.get('question', 'no question')
    documents_str = dp.get('documents', '[]')
    try:
        documents_list = ast.literal_eval(documents_str)
    except:
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
    if template_type in ['qwen']:
        prefix = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>"""
    elif template_type in ['llama']:
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
    return prefix
def gen_from_jsonl(path):
    """Load data from JSONL file and convert to dataset format"""
    with open(path) as f:
        for line in f:
            data = json.loads(line)    
            if 'supporting_facts' in data:
                evidence = []
                for fact in data['supporting_facts']:
                    title, sent_idx = fact
                    for doc in data['context']:
                        if doc[0] == title:
                            doc_text = " ".join(doc[1])
                            evidence.append(doc_text)
                            break
                data['evidences'] = str(evidence)
                data['supporting_facts'] = str(data['supporting_facts'])
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
def _has_valid_format(text: str) -> bool:
    """Check if text has valid <answer></answer> format"""
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
    """Check if answer is an IDK/uncertain expression"""
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
    """
    Call API to generate N candidate answers
    Returns: list of generated texts
    """
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests library is not available, cannot use API mode")
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
    if resp2.status_code != 200:
        print(f"\n⚠️  API error details:")
        print(f"  Status: {resp2.status_code}")
        print(f"  URL: {comp_url}")
        print(f"  Model: {model_name}")
        print(f"  Response: {resp2.text[:300]}")
        return []
    data2 = resp2.json()
    candidates = []
    for choice in data2.get('choices', []):
        text = choice.get('text', '').strip()
        if text:
            candidates.append(text)
    return candidates if candidates else []
def evaluate_sample_best_of_n(sample_dict, prompt, args, llm, sampling_params, postprocessor):
    """
    Evaluate single sample using Best-of-N strategy
    Returns: (is_truly_unanswerable: bool, best_reward: float)
    If any of the 32 inferences successfully answers (non-IDK and correct), returns False (not truly unanswerable)
    Only returns True (truly unanswerable) if all 32 attempts fail
    """
    import re
    extra_info = sample_dict.get('extra_info', {})
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except:
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
        if not candidates:
            print(f"  ⚠️ API call failed, skipping this sample")
            return False, -2
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
def create_answerable_samples(dataset, num_samples, template_type):
    """Create answerable samples"""
    print(f"\n{'='*80}")
    print(f"Step 1: Create {num_samples} ANSWERABLE samples")
    print(f"{'='*80}")
    answerable_dataset = dataset.select(range(num_samples))
    print(f"Selected {len(answerable_dataset)} samples for answerable")
    answerable_samples = []
    for i in range(len(answerable_dataset)):
        sample = answerable_dataset[i]
        answerable_sample = {
            'question': sample.get('question', ''),
            'documents': sample.get('documents', '[]'),
            'answer': sample['answer'],
            'data_source': sample.get('data_source', 'hotpot'),
            'evidences': sample.get('evidences', '[]'),
            'extra_info': sample.get('extra_info', {}),
        }
        if isinstance(answerable_sample['extra_info'], dict):
            answerable_sample['extra_info']['answerable'] = True
        else:
            answerable_sample['extra_info'] = {'answerable': True}
        answerable_samples.append(answerable_sample)
    print(f"✓ Created {len(answerable_samples)} answerable samples")
    return answerable_samples
def create_unanswerable_samples_with_filter(dataset, start_idx, num_samples, template_type, 
                                           args, llm, sampling_params, postprocessor):
    kept_samples = []
    removed_count = 0
    processed_count = 0
    max_samples_to_try = min(len(dataset) - start_idx, num_samples * 10)  
    pbar = tqdm(total=num_samples, desc="Filtering unanswerable samples")
    idx = start_idx
    while len(kept_samples) < num_samples and idx < len(dataset) - 1:
        question_sample = dataset[idx]
        question = question_sample.get('question', '')
        answer = question_sample['answer']
        try:
            original_documents = ast.literal_eval(question_sample.get('documents', '[]'))
        except:
            original_documents = []
        supporting_facts = question_sample.get('supporting_facts', None)
        if supporting_facts is None:
            extra_info = question_sample.get('extra_info', {})
            if isinstance(extra_info, str):
                try:
                    extra_info = ast.literal_eval(extra_info)
                except:
                    extra_info = {}
            supporting_facts = extra_info.get('supporting_facts', [])
        if isinstance(supporting_facts, str):
            try:
                supporting_facts = ast.literal_eval(supporting_facts)
            except:
                supporting_facts = []
        supporting_doc_titles = []
        if isinstance(supporting_facts, list):
            for fact in supporting_facts:
                if isinstance(fact, list) and len(fact) >= 1:
                    title = fact[0]
                    if title not in supporting_doc_titles:
                        supporting_doc_titles.append(title)
        if supporting_doc_titles and len(supporting_doc_titles) > 1 and original_documents:
            starting_node = supporting_doc_titles[0]
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
        evidences = question_sample.get('evidences', '[]')
        try:
            evidences_list = ast.literal_eval(evidences) if isinstance(evidences, str) else evidences
        except:
            evidences_list = []
        augmented_evidences = evidences_list
        unanswerable_sample = {
            'question': question,
            'documents': str(modified_documents),
            'answer': answer,
            'data_source': question_sample.get('data_source', 'hotpot'),
            'evidences': str(augmented_evidences),
            'extra_info': question_sample.get('extra_info', {}),
        }
        if isinstance(unanswerable_sample['extra_info'], dict):
            unanswerable_sample['extra_info']['answerable'] = False
        else:
            unanswerable_sample['extra_info'] = {'answerable': False}
        prompt = make_prefix_unified(unanswerable_sample, template_type)
        is_truly_unanswerable, best_reward = evaluate_sample_best_of_n(
            unanswerable_sample, prompt, args, llm, sampling_params, postprocessor
        )
        processed_count += 1
        if is_truly_unanswerable:
            kept_samples.append(unanswerable_sample)
            pbar.update(1)
        else:
            removed_count += 1
        idx += 1
        if idx >= len(dataset) - 1:
            break
    pbar.close()
    return kept_samples
def main():
    parser = argparse.ArgumentParser(description='Combined hotpot data processing and Best-of-N filtering')
    parser.add_argument('--type', type=str, default='train', help='train or test')
    parser.add_argument('--template_type', type=str, default='deepseek-r1-distill-qwen')
    parser.add_argument('--size', type=int, required=True, help='Total target sample size (split equally between answerable and unanswerable)')
    parser.add_argument('--data-path', type=str, default=None, help='Input JSONL file path')
    parser.add_argument('--model-path', type=str, default='', help='Local model path (vLLM mode)')
    parser.add_argument('--use-api', action='store_true', help='Use API mode instead of local vLLM')
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
    
    if args.data_path:
        data_path = args.data_path
    elif args.type == 'train':
        data_path = './hotpot/hotpot_train_v1.1.jsonl'
    else:
        data_path = './hotpot/hotpot_dev_distractor_v1.jsonl'
    
    answerable_size = args.size // 2
    unanswerable_size = args.size // 2
    
    print(f"\n📂 Loading data from {data_path}...")
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': data_path})
    print(f"   ✓ Raw dataset length: {len(raw_dataset)}")
    
    total_needed = answerable_size + unanswerable_size * 10  
    dataset = raw_dataset.shuffle(seed=42).select(range(min(total_needed, len(raw_dataset))))
    print(f"   ✓ Selected {len(dataset)} samples for processing")
    
    llm = None
    sampling_params = None
    
    if args.use_api:
        print(f"\n🌐 Testing API connection...")
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
        print(f"\n🔧 Loading model from {args.model_path}...")
        if not args.model_path:
            print("Error: --model-path parameter is required for local mode!")
            return
        
        try:
            from vllm import LLM, SamplingParams
            print("   ✓ vLLM module imported successfully")
        except ImportError:
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
    from verl.utils.reward_score.answer_postprocessor import get_postprocessor
    postprocessor = get_postprocessor()
    print("   ✓ Postprocessor initialized successfully")
    
    answerable_samples = create_answerable_samples(dataset, answerable_size, args.template_type)
    unanswerable_samples = create_unanswerable_samples_with_filter(
        dataset, answerable_size, unanswerable_size, args.template_type,
        args, llm, sampling_params, postprocessor
    )
    
    print(f"\n{'='*80}")
    print("Step 3: Merge and save dataset")
    print(f"{'='*80}")
    
    all_samples = answerable_samples + unanswerable_samples
    print(f"Total samples: {len(all_samples)}")
    print(f"  - Answerable: {len(answerable_samples)}")
    print(f"  - Unanswerable: {len(unanswerable_samples)}")
    
    combined_dataset = Dataset.from_list(all_samples)
    
    def regenerate_prompt(example, idx):
        question = make_prefix_unified(example, template_type=args.template_type)
        return {
            "prompt": question,
            "question": example['question'],
            "answer": example['answer'],
            "data_source": example['data_source'],
            "extra_info": example['extra_info'],
            "documents": example['documents'],
            "evidences": example['evidences'],
        }
    
    print("\nGenerating prompts...")
    combined_dataset = combined_dataset.map(regenerate_prompt, with_indices=True)
    
    combined_dataset = combined_dataset.shuffle(seed=42)
    
    output_dir = f'data/hotpot/{args.template_type}'
    os.makedirs(os.path.expanduser(output_dir), exist_ok=True)
    
    if args.type == 'train':
        output_file = os.path.join(output_dir, 'train.parquet')
    else:
        output_file = os.path.join(output_dir, 'test.parquet')
    
    combined_dataset.to_parquet(output_file)
    print(f"\n💾 Saved to {output_file}")
    print(f"   ✓ Final dataset: {len(combined_dataset)} samples")
    
    df_verify = pd.read_parquet(output_file)
    n_false = sum(1 for _, row in df_verify.iterrows() 
                  if isinstance(row.get('extra_info'), (dict, str)) and 
                  (json.loads(row['extra_info']) if isinstance(row['extra_info'], str) else row['extra_info']).get('answerable') == False)
    n_true = sum(1 for _, row in df_verify.iterrows() 
                 if isinstance(row.get('extra_info'), (dict, str)) and 
                 (json.loads(row['extra_info']) if isinstance(row['extra_info'], str) else row['extra_info']).get('answerable') == True)
    
    print(f"\nVerification:")
    print(f"   ✓ answerable=True: {n_true}")
    print(f"   ✓ answerable=False: {n_false}")
    
    if llm is not None:
        try:
            llm.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()