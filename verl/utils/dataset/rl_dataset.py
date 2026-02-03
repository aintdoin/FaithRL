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
from omegaconf import ListConfig
import os
from typing import List, Union
import copy
import pandas as pd
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}
    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)
    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)
    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)
    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output
class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=False,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 disable_unanswerable=None,
                 model_template=None):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]
        if model_template is None:
            model_template = os.environ.get('MODEL_TEMPLATE', 'qwen')
        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_overlong_prompts = filter_prompts
        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        # Configure whether to drop answerable=False rows
        if disable_unanswerable is None:
            try:
                _env = os.environ.get('disable_unanswerable', 'false').strip().lower()
                self.disable_unanswerable = _env in ('1', 'true', 'yes')
            except Exception:
                self.disable_unanswerable = False
        else:
            self.disable_unanswerable = bool(disable_unanswerable)
        self.model_template = model_template
        # Always enable system prompt injection
        from verl.utils.dataset.system_prompts import wrap_prompt_with_system
        self._wrap_prompt = wrap_prompt_with_system
        print(f"[RLHFDataset] Runtime system prompt injection enabled (template={model_template})")
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()
    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_local_path_from_hdfs
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        print(f'original dataset len: {len(self.dataframe)}')
        # Optional: filter out samples with answerable == False
        if self.disable_unanswerable:
            def _keep_row(row):
                extra = row.get('extra_info', None)
                ans_flag = True
                try:
                    if isinstance(extra, str):
                        try:
                            parsed = json.loads(extra)
                            extra = parsed if isinstance(parsed, dict) else {}
                        except Exception:
                            extra = {}
                    if not isinstance(extra, dict):
                        extra = {}
                    raw_val = extra.get('answerable', None)
                    if raw_val is None:
                        return True  # default keep
                    if isinstance(raw_val, bool):
                        ans_flag = raw_val
                    elif isinstance(raw_val, str):
                        v = raw_val.strip().lower()
                        if v in ('true', '1', 'yes'):
                            ans_flag = True
                        elif v in ('false', '0', 'no'):
                            ans_flag = False
                        else:
                            ans_flag = bool(v)
                    elif isinstance(raw_val, (int, np.integer)):
                        ans_flag = bool(raw_val)
                    else:
                        ans_flag = bool(raw_val)
                except Exception:
                    ans_flag = True
                return ans_flag is True
            before_len = len(self.dataframe)
            self.dataframe = self.dataframe[self.dataframe.apply(_keep_row, axis=1)]
            after_len = len(self.dataframe)
            print(f'filter disable_unanswerable=True: kept {after_len}/{before_len}')
        # nvm if prompt is too long
        # self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
        #     tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
        #                                                      axis=1)]
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            def is_prompt_length_valid(doc):
                try:
                    tokens = tokenizer(doc[prompt_key], truncation=False)['input_ids']
                    return len(tokens) <= self.max_prompt_length
                except Exception:
                    return True
            self.dataframe = self.dataframe[self.dataframe.apply(is_prompt_length_valid, axis=1)]
            print(f'filter dataset len: {len(self.dataframe)}')
    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()
        chat = row_dict.pop(self.prompt_key)
        prompt_with_chat_template = chat
        # Apply system prompt injection
        if self.model_template == 'llama':
             messages = self._wrap_prompt(prompt_with_chat_template, model_template='llama', return_list=True)
             # tokenize=True returns list of ints. add_generation_prompt=True adds the assistant start.
             prompt_with_chat_template = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
             # DEBUG: Print tokenization check
             if item == 0:  # Print only for the first item to avoid spam
                 print(f"\n[DEBUG] Llama Tokenization Check:")
                 print(f"Type of prompt: {type(prompt_with_chat_template)}")
                 print(f"First 20 tokens: {prompt_with_chat_template[:20]}")
                 decoded = self.tokenizer.decode(prompt_with_chat_template)
                 print(f"Decoded Prompt (Start): {decoded[:500]}...")
                 print(f"Decoded Prompt (End): ...{decoded[-200:]}")
        else:
            prompt_with_chat_template = self._wrap_prompt(
                prompt_with_chat_template, 
                model_template=self.model_template
            )
        # prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)
        position_ids = compute_position_id_with_mask(attention_mask)
        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()
        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        # Normalize extra_info and coerce answerable to bool (never None)
        extra_info = row_dict.get('extra_info', {})
        if isinstance(extra_info, str):
            try:
                parsed = json.loads(extra_info)
                extra_info = parsed if isinstance(parsed, dict) else {}
            except Exception:
                extra_info = {}
        try:
            if extra_info is None or (isinstance(extra_info, float) and pd.isna(extra_info)):
                extra_info = {}
        except Exception:
            pass
        if not isinstance(extra_info, dict):
            extra_info = {}
        # Coerce answerable
        if 'answerable' in extra_info:
            val = extra_info['answerable']
            if isinstance(val, bool):
                coerced = val
            elif isinstance(val, str):
                v = val.strip().lower()
                if v in ('true', '1', 'yes'):
                    coerced = True
                elif v in ('false', '0', 'no'):
                    coerced = False
                else:
                    coerced = bool(v)
            elif isinstance(val, (int, np.integer)):
                coerced = bool(val)
            else:
                coerced = bool(val)
            extra_info['answerable'] = coerced
        else:
            extra_info['answerable'] = True
        row_dict['extra_info'] = extra_info
        return row_dict
    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()