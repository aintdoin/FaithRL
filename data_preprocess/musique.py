""" Preprocess dataset for knights and knaves logic task """
import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import random
import threading
import concurrent.futures
def make_prefix(dp):
    question = dp.get('question', 'no question')
    documents_str = dp.get('documents', '[]')
    # Parse and format the documents.
    import ast
    try:
        documents_list = ast.literal_eval(documents_str)
    except:
        documents_list = []
    formatted_docs = []
    for doc in documents_list:
        if isinstance(doc, list) and len(doc) == 2:
            title, content = doc
            # content could be a list of paragraphs or a single string
            if isinstance(content, list):
                text = ' '.join(str(item) for item in content)
            else:
                text = str(content)
            formatted_docs.append(f"Document '{title}': {text}")
    documents_context = "\n".join(formatted_docs) if formatted_docs else "No references provided."
    # Define the user content block.
    user_content = f"""**References:**
{documents_context}
**Question:**
{question}"""
    return user_content
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--size', type=int, default=5000)
    parser.add_argument('--log-every', type=int, default=200, help='Print a status line every N records (0 to disable)')
    parser.add_argument('--verbose', action='store_true', help='Print per-record detailed status')
    args = parser.parse_args()
    # data_source = 'ASQA'
    if args.type == 'train':
        TRAIN_SIZE = args.size
        args.data_path = './musique/data/musique_full_v1.0_train.jsonl'
    else:
        TEST_SIZE = args.size
        args.data_path = './musique/data/musique_full_v1.0_dev.jsonl'
    args.hdfs_dir = None
    args.local_dir = 'data/musique'
    class QASentenceConverter:
        """Convert (question, answer, prior answers) into a single declarative sentence via LLM.
        Falls back to a deterministic template if API unavailable or fails.
        """
        def __init__(self):
            self.api_base = os.environ.get('LLM_JUDGE_API_BASE', '').strip()
            self.model_name = os.environ.get('LLM_JUDGE_MODEL_NAME', '').strip() or 'llm-judge'
            self.api_key = os.environ.get('LLM_JUDGE_API_KEY', '').strip()
            self.timeout = float(os.environ.get('LLM_JUDGE_TIMEOUT', '60'))
            self.max_workers = int(os.environ.get('LLM_JUDGE_MAX_WORKERS', '8'))
            try:
                import requests  # type: ignore
                self.requests = requests
            except Exception:
                self.requests = None
            self.use_api = bool(self.api_base and self.requests)
            # Stats
            self._lock = threading.Lock()
            self.stats = {
                'api_success': 0,
                'api_fallback': 0,
                'api_error': 0,
            }
        def _inc(self, key: str, value: int = 1) -> None:
            with self._lock:
                self.stats[key] = self.stats.get(key, 0) + value
        def _build_messages(self, question: str, answer: str, prior_answers: list) -> list:
            system_content = (
                "You are an expert at turning subquestion-answer pairs into one clear, natural English statement.\n\n"
                "Instructions:\n"
                "1. Produce ONE grammatically correct declarative sentence using the given subquestion and its answer.\n"
                "2. Preserve the semantic relationship between entities.\n"
                "3. If the subquestion contains references like #1, #2, etc., resolve them using the list of prior answers (1-indexed).\n"
                "4. Return ONLY the resulting sentence, nothing else.\n\n"
                "Examples:\n"
                "Q: who wrote crazy little thing called love original artist | A: Freddie Mercury\n"
                "→ Freddie Mercury wrote Crazy Little Thing Called Love.\n\n"
                "Q: In what year did #1 die? | A: 1991 | prior: ['Freddie Mercury']\n"
                "→ Freddie Mercury died in 1991."
            )
            user_content = (
                "Convert the following pair into a single statement.\n"
                f"Subquestion: {question}\n"
                f"Answer: {answer}\n"
                f"Prior answers (for # references): {json.dumps(prior_answers, ensure_ascii=False)}"
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
        def convert_one(self, question: str, answer: str, prior_answers: list) -> str:
            q = str(question or '').strip()
            a = str(answer or '').strip()
            if not q or not a:
                return ''
            if self.use_api:
                try:
                    messages = self._build_messages(q, a, prior_answers)
                    out = self._call_chat(messages)
                    if out:
                        self._inc('api_success', 1)
                        return out
                    self._inc('api_fallback', 1)
                except Exception:
                    self._inc('api_error', 1)
                    self._inc('api_fallback', 1)
            # Fallback template
            return f"{a} is the answer to: {q}."
    def load_examples_to_list(path, sample_size=None, seed=42):
        all_records = []
        with open(path) as f:
            for line in f:
                all_records.append(json.loads(line))
        if sample_size is not None and sample_size < len(all_records):
            random.seed(seed)
            selected_indices = random.sample(range(len(all_records)), sample_size)
            selected_records = [all_records[i] for i in selected_indices]
        else:
            selected_records = all_records
        print(f"Total records: {len(all_records)}, Selected: {len(selected_records)}")
        qa_converter = QASentenceConverter()
        log_every = args.log_every
        verbose = args.verbose
        records = []
        for idx_data, data in enumerate(tqdm(selected_records)):
            # Ensure data_source field exists
            if 'data_source' not in data:
                data['data_source'] = 'musique'
            if 'paragraphs' in data:
                paragraphs = data['paragraphs'] if isinstance(data['paragraphs'], list) else []
                documents = []
                for para in paragraphs:
                    title = para.get('title', '')
                    paragraph_text = para.get('paragraph_text', '')
                    documents.append([title, [paragraph_text]])
                data['documents'] = str(documents)
                del data['paragraphs']
            # Build evidences from question_decomposition ONLY for train split
            if args.type == 'train':
                evidences_list = []
                qd = data.get('question_decomposition')
                if isinstance(qd, list) and qd:
                    prior_answers = []
                    for qa_idx, qa in enumerate(qd):
                        q = (qa or {}).get('question', '')
                        a = (qa or {}).get('answer', '')
                        sent = qa_converter.convert_one(q, a, prior_answers)
                        if sent:
                            evidences_list.append(sent)
                        # Record current answer for future #n resolution
                        if isinstance(a, str) and a.strip():
                            prior_answers.append(a.strip())
                    # Remove the original decomposition as it's now summarized
                    del data['question_decomposition']
                else:
                    qd = qd if isinstance(qd, list) else []
                # Save evidences as string representation for consistency
                data['evidences'] = str(evidences_list)
                # Logging
                if verbose or (log_every > 0 and (idx_data + 1) % log_every == 0):
                    print(
                        f"[musique] idx={idx_data + 1}/{len(selected_records)} "
                        f"qd_items={(len(qd) if isinstance(qd, list) else 0)} "
                        f"evidences={len(evidences_list)} llm={qa_converter.stats['api_success']} "
                        f"fallback={qa_converter.stats['api_fallback']} api_errors={qa_converter.stats['api_error']}"
                    )
            else:
                # For test split, do not generate evidences; ensure empty and drop qd if present
                if 'question_decomposition' in data:
                    del data['question_decomposition']
                data['evidences'] = '[]'
            records.append(data)
        return records
    sample_size = TRAIN_SIZE if args.type == 'train' else TEST_SIZE
    raw_records = load_examples_to_list(args.data_path, sample_size=sample_size, seed=42)
    raw_dataset = Dataset.from_list(raw_records)
    print("raw_dataset length: ", len(raw_dataset))
    if args.type == 'train':
        train_dataset = raw_dataset
        print("train_dataset length: ", len(train_dataset))
    else:
        test_dataset = raw_dataset
        print("test_dataset length: ", len(test_dataset))
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = make_prefix(example)
            data_source = example.get('data_source', 'musique')
            answer = example.get('answer', '')
            answer_aliases = example.get('answer_aliases', [])
            sample_id = example.get('id')
            answerable = example.get('answerable', True)
            evidences = example.get('evidences', '[]')
            original_question = example.get('question', '')
            data = {
                "prompt": prompt,
                "question": original_question,
                "answer": answer,
                "data_source": data_source,
                "evidences": evidences,
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'sample_id': sample_id,
                    'answer_aliases': answer_aliases,
                    'answerable': answerable,
                }
            }
            return data
        return process_fn
    if args.type == 'train':
        # Remove original columns after they've been formatted into prompt
        # Keep 'documents' and 'evidences' for reference
        cols_to_remove = [col for col in train_dataset.column_names if col not in ['prompt', 'answer', 'data_source', 'extra_info', 'documents']]
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=cols_to_remove)
    else:
        # Remove original columns after they've been formatted into prompt
        # Keep 'documents' and 'evidences' for reference
        cols_to_remove = [col for col in test_dataset.column_names if col not in ['prompt', 'answer', 'data_source', 'extra_info', 'documents']]
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=cols_to_remove)
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)
    if args.type == 'train':
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    else:
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
# python -m examples.data_preprocess.musique_full