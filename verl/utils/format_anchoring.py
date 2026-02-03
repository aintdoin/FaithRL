import os
import torch
import numpy as np
from typing import List, Dict, Optional
import pyarrow.parquet as pq
import random
from dataclasses import dataclass

@dataclass
class FormatAnchorConfig:
    frequency: int = 50              
    steps_per_anchor: int = 2        
    lr_ratio: float = 0.1            
    batch_size: int = 16             
    data_start_idx: int = 3001       
    data_end_idx: int = 4000         
    format_check_strict: bool = True  
    verbose: bool = True             

class FormatAnchoringDataset:
    def __init__(
        self,
        data_file: str,  
        tokenizer,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.samples = []
        print(f"\n{'='*80}")
        print(f"📋 ")
        print(f"{'='*80}")
        print(f"  File: {data_file}")
        self._load_dataset(data_file, max_samples)
        print(f"\n✓ Loaded {len(self.samples)} samples")
        print(f"{'='*80}\n")

    def _load_dataset(self, file_path: str, max_samples: Optional[int]):
        try:
            if not os.path.exists(file_path):
                print(f"  ✗ File not found: {file_path}")
                return
            table = pq.read_table(file_path)
            df = table.to_pandas()
            total_rows = len(df)
            print(f"  ✓ Dataset size: {total_rows} samples")
            if max_samples is not None and max_samples < total_rows:
                print(f"  ℹ️  Load only first {max_samples} samples")
                df = df.iloc[:max_samples]
            loaded_count = 0
            for idx, row in df.iterrows():
                prompt = row.get('prompt', '')
                response = row.get('response', '')
                if not prompt or not response:
                    continue
                if self._is_format_valid(response):
                    self.samples.append({
                        'prompt': prompt,
                        'response': response,
                        'dataset': row.get('data_source', 'unknown'),
                        'original_idx': row.get('original_idx', idx),
                        'question': row.get('question', ''),
                        'answer': row.get('answer', ''),
                    })
                    loaded_count += 1
            from collections import Counter
            source_counts = Counter(s['dataset'] for s in self.samples)
            for source, count in source_counts.items():
                print(f"    {source}: {count} samples")
            print(f"  ✓ Successfully loaded {loaded_count} valid samples")
        except Exception as e:
            print(f"  ✗ Load failed: {e}")
            import traceback
            traceback.print_exc()

    def _is_format_valid(self, response: str) -> bool:
        if '<answer>' not in response.lower() or '</answer>' not in response.lower():
            return False
        has_think = '' in response.lower()
        if has_think != has_think_end:
            return False
        return True

    def sample_batch(self, batch_size: int) -> List[Dict]:
        if len(self.samples) < batch_size:
            return random.choices(self.samples, k=batch_size)
        else:
            return random.sample(self.samples, batch_size)

    def __len__(self):
        return len(self.samples)

class FormatAnchor:
    def __init__(
        self,
        config: FormatAnchorConfig,
        tokenizer,
        data_file: str,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = FormatAnchoringDataset(
            data_file=data_file,
            tokenizer=tokenizer,
            max_samples=None,  
        )
        self.total_anchors = 0
        self.anchor_history = []

    def should_anchor(self, global_step: int) -> bool:
        if global_step == 0:
            return False
        return global_step % self.config.frequency == 0

    def anchor(
        self,
        actor_module,
        optimizer,
        device='cuda'
    ) -> Dict[str, float]:
        if len(self.dataset) == 0:
            print("⚠️  Warning: No format anchoring data available")
            return {'anchor_loss': 0.0, 'samples': 0}
        
        original_lrs = [pg['lr'] for pg in optimizer.param_groups]
        anchor_lr = original_lrs[0] * self.config.lr_ratio
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = anchor_lr
        
        actor_module.train()
        total_loss = 0.0
        num_samples = 0
        
        if self.config.verbose:
            print(f"\n{'─'*60}")
            print(f"🔧  (LR: {anchor_lr:.2e})")
        
        for step in range(self.config.steps_per_anchor):
            batch_samples = self.dataset.sample_batch(self.config.batch_size)
            batch_data = self._prepare_batch(batch_samples, device)
            loss = self._compute_sft_loss(actor_module, batch_data)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_module.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_samples += len(batch_samples)
            
            if self.config.verbose:
                print(f"  Step {step+1}/{self.config.steps_per_anchor}: loss={loss.item():.4f}")
        
        for param_group, original_lr in zip(optimizer.param_groups, original_lrs):
            param_group['lr'] = original_lr
        
        avg_loss = total_loss / self.config.steps_per_anchor
        self.total_anchors += 1
        self.anchor_history.append({
            'step': self.total_anchors * self.config.frequency,
            'loss': avg_loss,
            'samples': num_samples
        })
        
        if self.config.verbose:
            print(f"  ✓ : avg loss={avg_loss:.4f}, samples={num_samples}")
            print(f"{'─'*60}\n")
        
        return {
            'anchor_loss': avg_loss,
            'anchor_samples': num_samples,
            'total_anchors': self.total_anchors
        }

    def _prepare_batch(self, batch_samples: List[Dict], device) -> Dict:
        prompts = [sample['prompt'] for sample in batch_samples]
        responses = [sample['response'] for sample in batch_samples]
        full_texts = [p + r for p, r in zip(prompts, responses)]
        
        encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        labels = input_ids.clone()
        
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            prompt_length = len(prompt_tokens)
            labels[i, :prompt_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def _compute_sft_loss(self, model, batch_data: Dict) -> torch.Tensor:
        input_ids = batch_data['input_ids']
        labels = batch_data['labels']
        attention_mask = batch_data['attention_mask']
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            return outputs.loss
        
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss

    def get_statistics(self) -> Dict:
        if len(self.anchor_history) == 0:
            return {
                'total_anchors': 0,
                'avg_loss': 0.0,
                'latest_loss': 0.0
            }
        
        avg_loss = np.mean([h['loss'] for h in self.anchor_history])
        latest_loss = self.anchor_history[-1]['loss']
        
        return {
            'total_anchors': self.total_anchors,
            'avg_loss': avg_loss,
            'latest_loss': latest_loss,
            'history': self.anchor_history
        }

def integrate_format_anchoring(trainer_instance, config: FormatAnchorConfig, data_file: str):
    format_anchor = FormatAnchor(
        config=config,
        tokenizer=trainer_instance.tokenizer,
        data_file=data_file
    )
    trainer_instance.format_anchor = format_anchor
    return trainer_instance