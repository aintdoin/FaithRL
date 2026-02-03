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
"""
The main entry point to run the PPO algorithm
"""
import logging
import os
import re
import ast
import nltk
import warnings
import numpy as np
import functools
import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, offload_fsdp_grad, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, \
    load_fsdp_param_and_grad
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from codetiming import Timer
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    else:
        raise ValueError(
            'HSDP is not supported yet because it produces incorrect results for now. Please set fsdp_size=-1')
        assert world_size % fsdp_size == 0
        device_mesh = init_device_mesh('cuda',
                                       mesh_shape=(world_size // fsdp_size, fsdp_size),
                                       mesh_dim_names=['ddp', 'fsdp'])
    return device_mesh
def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy
class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """
    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)
        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']
        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']
        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)
        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= (self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size)
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (self.device_mesh.shape[0] //
                                                            self.ulysses_sequence_parallel_size)
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0
        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                               self.ulysses_sequence_parallel_size)
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                           self.ulysses_sequence_parallel_size)
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               use_remove_padding=False,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False,
                               use_liger=False,
                               role='actor'):
        from verl.utils.model import print_model_size, update_model_config, get_generation_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload
        from torch import optim
        assert role in ['actor', 'ref']
        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)
        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)
        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(actor_model_config, verbose=True)
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')
        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code)
            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=actor_module)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)
            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(actor_module)
        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)
        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))
        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None
        print(f'wrap_policy: {auto_wrap_policy}')
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == 'actor' else CPUOffload(offload_params=True)
        actor_module_fsdp = FSDP(
            actor_module,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False)
        log_gpu_memory_usage('After Actor FSDP init', logger=logger)
        # TODO: add more optimizer args into config
        if role == 'actor':
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))
            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')
            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None
        log_gpu_memory_usage('After actor optimizer init', logger=logger)
        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config
    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])
        if self.config.rollout.name == 'hf':
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager import BaseShardingManager
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            log_gpu_memory_usage('Before building vllm rollout', logger=None)
            local_path = copy_local_path_from_hdfs(self.config.model.path)
            if vllm_mode == 'customized':
                rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                      config=self.config.rollout,
                                      tokenizer=self.tokenizer,
                                      model_hf_config=self.actor_model_config)
            elif vllm_mode == 'spmd':
                rollout = vLLMRollout(model_path=local_path,
                                      config=self.config.rollout,
                                      tokenizer=self.tokenizer,
                                      model_hf_config=self.actor_model_config,
                                      device_mesh=rollout_device_mesh)
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")
            log_gpu_memory_usage('After building vllm rollout', logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            rollout_sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                               inference_engine=rollout.inference_engine,
                                                               model_config=self.actor_model_config,
                                                               full_params='hf' in self.config.rollout.load_format,
                                                               device_mesh=rollout_device_mesh)
            log_gpu_memory_usage('After building sharding manager', logger=None)
        return rollout, rollout_sharding_manager
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        use_remove_padding = self.config.model.get('use_remove_padding', False)
        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False),
                use_liger=self.config.model.get('use_liger', False),
                role='actor')
            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module
            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)
        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()
        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               use_remove_padding=use_remove_padding,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False),
                                                               use_liger=self.config.model.get('use_liger', False),
                                                               role='ref')[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(model=self.actor_module_fsdp,
                                                            optimizer=self.actor.actor_optimizer,
                                                            lr_scheduler=self.actor_lr_scheduler,
                                                            tokenizer=self.tokenizer)
        torch.cuda.empty_cache()
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to('cuda')
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())
        data.batch = data.batch.cuda()
        log_gpu_memory_usage('Before update policy', logger=logger)
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name='update_policy', logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/lr'] = lr
            log_gpu_memory_usage('After update policy', logger=logger)
            # TODO: here, we should return all metrics
            # FLOPs accounting is returned via meta_info['flops_step'] (raw FLOPs, global-batch scope).
            # NOTE: data.meta_info is shared across DP ranks (not chunked), so do NOT all-reduce here.
            try:
                actor_update_flops = float(self.flops_counter.estimate_total_flops(global_num_tokens))
            except Exception:
                actor_update_flops = 0.0
            output = DataProto(meta_info={'metrics': metrics, 'flops_step': actor_update_flops})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def format_anchor(self, format_anchor_config, format_anchor_dataset):
        assert self._is_actor, "Format anchoring only works on actor"
        # Load model and optimizer if offloaded
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                    device_id=torch.cuda.current_device(),
                                    load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())
        # Save original learning rate
        original_lrs = [pg['lr'] for pg in self.actor_optimizer.param_groups]
        # Set format anchoring learning rate (lower than main training)
        anchor_lr = original_lrs[0] * format_anchor_config.lr_ratio
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = anchor_lr
        # Set model to training mode
        self.actor_module_fsdp.train()
        total_loss = 0.0
        num_samples = 0
        if format_anchor_config.verbose and torch.distributed.get_rank() == 0:
            print(f"\n{'─'*60}")
        import random
        for step in range(format_anchor_config.steps_per_anchor):
            # Sample a batch from format_anchor_dataset
            if len(format_anchor_dataset) < format_anchor_config.batch_size:
                batch_samples = random.choices(format_anchor_dataset, k=format_anchor_config.batch_size)
            else:
                batch_samples = random.sample(format_anchor_dataset, format_anchor_config.batch_size)
            # Prepare batch data
            prompts = [sample['prompt'] for sample in batch_samples]
            responses = [sample['response'] for sample in batch_samples]
            full_texts = [p + r for p, r in zip(prompts, responses)]
            # Tokenize
            encodings = self.tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].cuda()
            attention_mask = encodings['attention_mask'].cuda()
            # Create labels (only compute loss on response part)
            labels = input_ids.clone()
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
                prompt_length = len(prompt_tokens)
                labels[i, :prompt_length] = -100  # Ignore prompt in loss
            # Forward pass
            outputs = self.actor_module_fsdp(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else self._compute_sft_loss(outputs.logits, labels)
            # Backward pass
            self.actor_optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor_module_fsdp.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            total_loss += loss.item()
            num_samples += len(batch_samples)
            if format_anchor_config.verbose and torch.distributed.get_rank() == 0:
                print(f"  Step {step+1}/{format_anchor_config.steps_per_anchor}: loss={loss.item():.4f}")
        # Restore original learning rate
        for param_group, original_lr in zip(self.actor_optimizer.param_groups, original_lrs):
            param_group['lr'] = original_lr
        avg_loss = total_loss / format_anchor_config.steps_per_anchor
        if format_anchor_config.verbose and torch.distributed.get_rank() == 0:
            print(f"{'─'*60}\n")
        # Offload if needed
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        # Return metrics
        metrics = {
            'anchor_loss': avg_loss,
            'anchor_samples': num_samples,
        }
        return DataProto(meta_info=metrics)
    def _compute_sft_loss(self, logits, labels):
        """Compute SFT loss manually if model doesn't return it"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to(torch.cuda.current_device())
        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        meta_info = {
            'eos_token_id':
                self.generation_config.eos_token_id
                if self.generation_config is not None else self.tokenizer.eos_token_id,
            'pad_token_id':
                self.generation_config.pad_token_id
                if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            log_gpu_memory_usage('After rollout generation', logger=logger)
            output = self.rollout_sharding_manager.postprocess_data(output)
        # FLOPs accounting for rollout generation (server-side-ish via vLLM):
        # Use a conservative forward-linear model on total (prompt+response) valid tokens to avoid overcounting KV-cache.
        try:
            attn = output.batch['attention_mask']
            tokens_sum = int(attn.sum().detach().item())
            gen_flops = float(self.flops_counter.estimate_total_flops_forward_linear(tokens_sum))
        except Exception:
            gen_flops = 0.0
        output.meta_info = dict(output.meta_info or {})
        output.meta_info['flops_step'] = gen_flops
        output = output.to('cpu')
        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        data = data.to('cuda')
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info['temperature'] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'old_log_probs': output},
                                         meta_info={'temperature': self.config.rollout.temperature})
            output = self.ulysses_sharding_manager.postprocess_data(output)
        # FLOPs accounting (forward-only): recompute log-prob is a forward pass.
        try:
            attn = data.batch['attention_mask']
            batch_seqlens = attn.sum(-1).detach().to('cpu').tolist()
            lp_flops = float(self.flops_counter.estimate_total_flops_forward(batch_seqlens))
        except Exception:
            lp_flops = 0.0
        output.meta_info = dict(output.meta_info or {})
        output.meta_info['flops_step'] = lp_flops
        output = output.to('cpu')
        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)
        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After compute_log_prob', logger=logger)
        return output
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        data = data.to('cuda')
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = self.ulysses_sharding_manager.postprocess_data(output)
        # FLOPs accounting (forward-only): ref log-prob is a forward pass.
        try:
            attn = data.batch['attention_mask']
            batch_seqlens = attn.sum(-1).detach().to('cpu').tolist()
            ref_lp_flops = float(self.flops_counter.estimate_total_flops_forward(batch_seqlens))
        except Exception:
            ref_lp_flops = 0.0
        output.meta_info = dict(output.meta_info or {})
        output.meta_info['flops_step'] = ref_lp_flops
        output = output.to('cpu')
        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_policy.actor_module._handle.reshard(True)
        torch.cuda.empty_cache()
        return output
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, remove_previous_ckpt=False):
        # only support save and load ckpt for actor
        assert self._is_actor
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                remove_previous_ckpt=remove_previous_ckpt)
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=False):
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
class CriticWorker(Worker):
    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config
        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload
        # normalize config
        self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() //
                                                  self.ulysses_sequence_parallel_size)
            self.config.forward_micro_batch_size //= (torch.distributed.get_world_size() //
                                                      self.ulysses_sequence_parallel_size)
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0
    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim
        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.
        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))
        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')
        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        from transformers import AutoConfig, AutoModelForTokenClassification
        from torch import nn
        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1
        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(critic_model_config.model_type)
        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(critic_model_config, verbose=True)
        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, 'classifier_dropout', 0.)
            setattr(critic_model_config, 'hidden_dropout', '0')
            critic_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            torch_dtype=torch_dtype,
                                                                            config=critic_model_config,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)
            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)
            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if self.rank == 0:
            print_model_size(critic_module)
        self.critic_model_config = critic_model_config
        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)
        log_gpu_memory_usage('Before critic FSDP', logger=None)
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=False,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=sharding_strategy,
                             mixed_precision=mixed_precision,
                             sync_module_states=True,
                             forward_prefetch=False,
                             device_mesh=self.device_mesh,
                             cpu_offload=None)
        log_gpu_memory_usage('After critic FSDP', logger=None)
        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))
        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')
        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)
        return critic_module, critic_optimizer, critic_lr_scheduler
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(model=self.critic_module,
                                                        optimizer=self.critic_optimizer,
                                                        lr_scheduler=self.critic_lr_scheduler,
                                                        tokenizer=self.tokenizer)
        torch.cuda.empty_cache()
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
        # FLOPs accounting (forward-only): value computation is a forward pass.
        try:
            attn = data.batch['attention_mask']
            batch_seqlens = attn.sum(-1).detach().to('cpu').tolist()
            v_flops = float(self.flops_counter.estimate_total_flops_forward(batch_seqlens))
        except Exception:
            v_flops = 0.0
        output.meta_info = dict(output.meta_info or {})
        output.meta_info['flops_step'] = v_flops
        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name='update_critic', logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics['critic/lr'] = lr
            try:
                critic_update_flops = float(self.flops_counter.estimate_total_flops(global_num_tokens))
            except Exception:
                critic_update_flops = 0.0
            output = DataProto(batch=None, meta_info={'metrics': metrics, 'flops_step': critic_update_flops})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, remove_previous_ckpt=False):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                remove_previous_ckpt=remove_previous_ckpt)
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=True):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """
    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config
        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size
    def _build_model(self, config):
        """
        Initialize MiniCheck API client instead of loading local model
        """
        import os
        import requests
        # Get MiniCheck API configuration from environment
        self.minicheck_api_base = os.environ.get('MINICHECK_API_BASE', '').strip()
        self.minicheck_use_api = bool(self.minicheck_api_base)
        self.minicheck_max_workers = int(os.environ.get('MINICHECK_MAX_WORKERS', '8'))
        self.minicheck_timeout = int(os.environ.get('MINICHECK_TIMEOUT', '30'))
        # Initialize tokenizer for token counting
        if self.config.model.input_tokenizer is not None:
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.model.get('trust_remote_code', False))
        else:
            # Use a default tokenizer for token counting if not specified
            from transformers import AutoTokenizer
            self.input_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # Test API connection only when explicitly configured
        if self.minicheck_use_api:
            try:
                response = requests.get(f"{self.minicheck_api_base}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"MiniCheck API is available at {self.minicheck_api_base}")
                else:
                    logger.warning(f"MiniCheck API health check failed: {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to connect to MiniCheck API at {self.minicheck_api_base}: {e}")
                logger.error("Please ensure the MiniCheck server is running using start_minicheck_server.sh")
        else:
            logger.info("MiniCheck API disabled by default (set MINICHECK_API_BASE to enable)")
        # Return None as we're using API instead of local model
        return None
    def _call_minicheck_api(self, docs, claims):
        """
        Call MiniCheck API to score claims against documents
        Args:
            docs: List of documents
            claims: List of claims
        Returns:
            List of probabilities (raw_probs from MiniCheck)
        """
        import requests
        import json
        # If API not configured, return neutral scores
        if not getattr(self, 'minicheck_use_api', False):
            return [0.5] * len(docs)
        try:
            payload = {
                "docs": docs,
                "claims": claims,
                "chunk_size": 32768
            }
            response = requests.post(
                f"{self.minicheck_api_base}/score",
                json=payload,
                timeout=self.minicheck_timeout
            )
            if response.status_code == 200:
                result = response.json()
                return result['raw_probs']
            else:
                logger.error(f"MiniCheck API error: {response.status_code} - {response.text}")
                # Return default scores on error
                return [0.5] * len(docs)
        except Exception as e:
            logger.error(f"Error calling MiniCheck API: {e}")
            # Return default scores on error
            return [0.5] * len(docs)
    def _predict_minicheck(self, doc_claim_pairs):
        """
        Predict scores for document-claim pairs using MiniCheck API
        Args:
            doc_claim_pairs: List of (doc, claim) tuples
        Returns:
            List of torch tensors with probabilities
        """
        if not doc_claim_pairs:
            return []
        docs = [pair[0] for pair in doc_claim_pairs]
        claims = [pair[1] for pair in doc_claim_pairs]
        # Call API
        probs = self._call_minicheck_api(docs, claims)
        # Convert to torch tensors to match HHEM interface
        return [torch.tensor(prob, dtype=torch.float32) for prob in probs]
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
    def _select_rm_score_fn(self, data_source):
        from verl.utils.reward_score import gsm8k, math, multiply, countdown, kk, halueval, hotpot
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
        elif data_source == 'halueval':
            return halueval.compute_score
        elif data_source == 'ASQA':
            return asqa.compute_score
        elif data_source in ['hotpot', '2wikimultihop', 'musique_ans', 'musique']:
            return hotpot.compute_score
        else:
            raise NotImplementedError
    def extract_solution(self, solution_str: str) -> str:
        """Extracts the final answer from the model's response string.
        Args:
            solution_str: Raw response string from the language model
        Returns:
            only response string
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
            return solution_str
        return processed_str.strip()
    def validate_model_reasoning_documents_only(self, documents, output_str, valid_response_ids=None, positive_score=1.0, negative_score=-1.0):
        """
        Improved version using stepwise logic for better alignment and segmentation,
        but evaluating strictly against documents (using documents as evidences).
        Enforces binary mask: only positive_score or negative_score (no 0.0).
        """
        reasoning_pattern = r'<think>(.*?)</think>'
        response = output_str
        # CRITICAL: Token Alignment (Adapted from stepwise)
        if valid_response_ids is not None:
            num_tokens = len(valid_response_ids)
            token_offsets = []
            char_pos = 0
            for token_id in valid_response_ids:
                token_text = self.input_tokenizer.decode([token_id])
                token_len = len(token_text)
                token_offsets.append((char_pos, char_pos + token_len))
                char_pos += token_len
        else:
            # Fallback not recommended for precise alignment
            return -2, []
        # Initialize with negative_score (guilty until proven innocent) to avoid 0.0
        sentence_mask = [negative_score] * num_tokens
        def set_span_value(char_start, char_end, value):
            for ti, (ts, te) in enumerate(token_offsets):
                if ts >= char_start and ts < char_end:
                    sentence_mask[ti] = value
        import re
        import difflib
        import numpy as np
        think_matches = list(re.finditer(reasoning_pattern, response, re.DOTALL))
        if not think_matches:
            return -2, sentence_mask
        think_match = think_matches[-1]
        t_content_start = think_match.start(1)
        t_content_end = think_match.end(1)
        reasoning_str = response[t_content_start:t_content_end]
        # Use enumeration pattern for step segmentation (better than simple sent_tokenize)
        enum_pattern = re.compile(r'(?m)^(\s*)(\d+)\.(\s*)')
        markers = list(enum_pattern.finditer(reasoning_str))
        n_marker = len(markers)
        last_end = 0
        seen_segments = []
        step_wise_scores = []
        for i, m in enumerate(markers):
            num_rel_start = m.start(2)
            num_rel_end = m.end(2)
            dot_rel_end = m.end(2) + 1
            num_global_start = t_content_start + num_rel_start
            dot_global_end = t_content_start + dot_rel_end
            # Mark the number itself as positive (encouraging structure)
            set_span_value(num_global_start, dot_global_end, positive_score)
            seg_rel_start = dot_rel_end
            if i + 1 < n_marker:
                seg_rel_end = markers[i+1].start(0)
            else:
                seg_rel_end = len(reasoning_str)
            segment_text = reasoning_str[seg_rel_start:seg_rel_end]
            global_start = t_content_start + seg_rel_start
            global_end = t_content_start + seg_rel_end
            # Repetition check
            is_repeated = False
            if segment_text.strip() and len(seen_segments) > 0:
                last_seen = seen_segments[-1]
                if difflib.SequenceMatcher(None, segment_text, last_seen).ratio() > 0.8:
                    is_repeated = True
                if not is_repeated:
                    matcher = difflib.SequenceMatcher(None, segment_text, last_seen)
                    match = matcher.find_longest_match(0, len(segment_text), 0, len(last_seen))
                    if len(segment_text) > 10 and match.size / len(segment_text) > 0.8:
                        is_repeated = True
            score_val = negative_score
            if is_repeated:
                score_val = negative_score
            elif not segment_text.strip():
                last_end = seg_rel_end
                continue
            else:
                # Evaluate using documents as evidences
                try:
                    judge_raw = self._evaluate_reasoning_segment(
                        segment_text=segment_text,
                        documents=documents,
                        evidences=documents, # KEY: Use documents as evidences
                        question=None,
                        ground_truth=None
                    )
                    if judge_raw is None:
                        score_val = negative_score
                    elif judge_raw > 0.5:
                        score_val = positive_score
                    else:
                        score_val = negative_score
                    seen_segments.append(segment_text)
                except Exception:
                    score_val = negative_score
            set_span_value(global_start, global_end, score_val)
            step_wise_scores.append(score_val)
            last_end = seg_rel_end
        # Handle remaining text (if any) or if no markers found
        if n_marker == 0 or last_end < len(reasoning_str):
            seg_rel_start = last_end
            seg_rel_end = len(reasoning_str)
            segment_text = reasoning_str[seg_rel_start:seg_rel_end]
            global_start = t_content_start + seg_rel_start
            global_end = t_content_start + seg_rel_end
            if segment_text.strip():
                try:
                    judge_raw = self._evaluate_reasoning_segment(
                        segment_text=segment_text,
                        documents=documents,
                        evidences=documents,
                        question=None,
                        ground_truth=None
                    )
                    if judge_raw is None:
                        score_val = negative_score
                    elif judge_raw > 0.5:
                        score_val = positive_score
                    else:
                        score_val = negative_score
                except Exception:
                    score_val = negative_score
                set_span_value(global_start, global_end, score_val)
                step_wise_scores.append(score_val)
        if not step_wise_scores:
            reasoning_score = negative_score
        else:
            reasoning_score = np.mean(step_wise_scores)
        return reasoning_score, sentence_mask
    def validate_model_reasoning_stepwise(self, output_str, documents, evidences, valid_response_ids=None, question=None, ground_truth=None, answer_aliases=None, answerable=True):
        reasoning_pattern = r'<think>(.*?)</think>'
        answer_pattern = r'<answer>(.*?)</answer>'
        response = output_str
        eos_tag = ' '
        judge_mode = os.environ.get("EVAR_REASONING_JUDGE_MODE", "llm").strip().lower()
        skip_llm_judge = judge_mode in ("rule", "heuristic", "none", "off", "false", "0")
        # CRITICAL FIX: Ensure token alignment with valid_response_ids
        if valid_response_ids is not None:
            # Use the actual response token IDs to ensure perfect alignment
            num_tokens = len(valid_response_ids)
            token_offsets = []
            char_pos = 0
            for token_id in valid_response_ids:
                token_text = self.input_tokenizer.decode([token_id])
                token_len = len(token_text)
                token_offsets.append((char_pos, char_pos + token_len))
                char_pos += token_len
        else:
            # Fallback: re-tokenize (may cause alignment issues!)
            enc = self.input_tokenizer(
                response,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            offsets = enc.get('offset_mapping')
            if isinstance(offsets[0], tuple) or isinstance(offsets[0], list):
                token_offsets = offsets
            else:
                token_offsets = offsets[0]
            num_tokens = len(token_offsets)
        sentence_mask = [0.0] * num_tokens  
        def set_span_value(char_start: int, char_end: int, value: float):
            for ti, (ts, te) in enumerate(token_offsets):
                if ts >= char_start and ts < char_end:
                    sentence_mask[ti] = value
        step_wise_scores = []
        reasoning_score = 0.0
        seen_segments = []
        import re
        import difflib
        think_matches = list(re.finditer(reasoning_pattern, response, re.DOTALL))
        think_match = think_matches[-1] if len(think_matches) > 0 else None
        if think_match is not None:
            t_content_start = think_match.start(1)
            t_content_end = think_match.end(1)
            reasoning_str = response[t_content_start:t_content_end]
            enum_pattern = re.compile(r'(?m)^(\s*)(\d+)\.(\s*)')
            markers = list(enum_pattern.finditer(reasoning_str))
            n_marker = len(markers)
            last_end = 0
            segments_meta = []  # [{'text': str, 'global_start': int, 'global_end': int, 'forced': Optional[float]}]
            def _append_segment(seg_text: str, g_start: int, g_end: int, *, forced: float = None):
                st = (seg_text or "").strip()
                if not st:
                    return
                segments_meta.append({
                    'text': st,
                    'global_start': int(g_start),
                    'global_end': int(g_end),
                    'forced': forced,
                })
            for i, m in enumerate(markers):
                num_rel_start = m.start(2)
                num_rel_end = m.end(2)
                dot_rel_end = m.end(2) + 1
                num_global_start = t_content_start + num_rel_start
                dot_global_end = t_content_start + dot_rel_end
                set_span_value(num_global_start, dot_global_end, 1.0)
                seg_rel_start = dot_rel_end
                if i + 1 < n_marker:
                    seg_rel_end = markers[i+1].start(0)
                else:
                    seg_rel_end = len(reasoning_str)
                segment_text = reasoning_str[seg_rel_start:seg_rel_end]
                global_start = t_content_start + seg_rel_start
                global_end = t_content_start + seg_rel_end
                # Check for repetition
                is_repeated = False
                if segment_text.strip() and len(seen_segments) > 0:
                    last_seen = seen_segments[-1]
                    # 1. Difflib ratio check against previous segment only
                    if difflib.SequenceMatcher(None, segment_text, last_seen).ratio() > 0.8:
                        is_repeated = True
                    # 2. Rule-based LCS (Longest Common Substring) check
                    if not is_repeated:
                        # Use difflib to find longest common substring
                        matcher = difflib.SequenceMatcher(None, segment_text, last_seen)
                        match = matcher.find_longest_match(0, len(segment_text), 0, len(last_seen))
                        # If the longest common substring covers most of the current segment
                        if len(segment_text) > 10 and match.size / len(segment_text) > 0.8:
                            is_repeated = True
                if is_repeated:
                    # HEAVY PENALTY for repetition loops to force stop or change
                    _append_segment(segment_text, global_start, global_end, forced=-1.0)
                elif not segment_text.strip():
                    last_end = seg_rel_end
                    continue
                else:
                    # Defer judging: collect for a single batched call (K <= 5).
                    _append_segment(segment_text, global_start, global_end, forced=None)
                    seen_segments.append(segment_text)
                last_end = seg_rel_end
            if n_marker == 0 or last_end < len(reasoning_str):
                seg_rel_start = last_end
                seg_rel_end = len(reasoning_str)
                segment_text = reasoning_str[seg_rel_start:seg_rel_end]
                global_start = t_content_start + seg_rel_start
                global_end = t_content_start + seg_rel_end
                if segment_text.strip():
                    _append_segment(segment_text, global_start, global_end, forced=None)
            # ===== Single-call judging for all segments (reduce FLOPs & overhead) =====
            if segments_meta:
                # 1) Handle IDK segments locally (avoid extra judge calls unless likely IDK)
                for seg in segments_meta:
                    if seg.get('forced') is not None:
                        continue
                    try:
                        has_idk = self._check_insufficient_info(seg.get('text', ''), question=question)
                    except Exception:
                        has_idk = None
                    if has_idk is True and answerable is False:
                        seg['forced'] = 1.0
                    elif has_idk is True and answerable is True:
                        seg['forced'] = -1.0
                # 2) Judge remaining segments in one request (expects JSON list of 0/1)
                judged = None
                if skip_llm_judge:
                    judged = [1.0 for _ in segments_meta]
                else:
                    try:
                        from verl.utils.reward_score.answer_postprocessor import get_postprocessor
                        post = get_postprocessor()
                        if getattr(post, 'use_judge_api', False):
                            # Build evidence text (consistent with _evaluate_reasoning_segment)
                            evd = evidences
                            if isinstance(evidences, str):
                                try:
                                    evd = ast.literal_eval(evidences)
                                except Exception:
                                    evd = evidences
                            evd_list = evd if isinstance(evd, (list, tuple)) else [evd]
                            evd_texts = []
                            for x in evd_list:
                                try:
                                    t = str(x)
                                except Exception:
                                    t = ''
                                if t:
                                    evd_texts.append(t)
                            max_evd = 8
                            evd_text = '\n'.join([f"- {t}" for t in evd_texts[:max_evd]]) if evd_texts else ""
                            prefix_lines = []
                            prefix_lines.append("You are a strict reasoning consistency judge.")
                            prefix_lines.append("Task: For EACH segment below, output 1 if it is FULLY SUPPORTED by the evidences; otherwise output 0.")
                            prefix_lines.append("Rules:")
                            prefix_lines.append("1) Output ONLY valid JSON: a list of 0/1 with the same length and order as the segments.")
                            prefix_lines.append("2) Be strict: if a segment is vague, adds unsupported facts, or is not grounded in evidences, output 0.")
                            prefix_lines.append("3) Do not output any explanation or extra text.")
                            if evd_text:
                                prefix_lines.append("")
                                prefix_lines.append(f"Evidences:\n{evd_text}")
                            prefix_lines.append("")
                            prefix_lines.append("Segments:")
                            prefix = "\n".join(prefix_lines) + "\n"
                            seg_lines = []
                            for idx, seg in enumerate(segments_meta, start=1):
                                seg_lines.append(f"{idx}. {seg.get('text','')}")
                            suffix = "\n".join(seg_lines) + "\n\nOutput JSON list:"
                            judge_prompt = prefix + suffix
                            import hashlib as _hashlib
                            _cache_key = "reasoning_steps_batch_prefix:" + _hashlib.sha1(prefix.encode("utf-8", errors="ignore")).hexdigest()
                            raw = post._call_judge_api(
                                judge_prompt,
                                cache_prefix=prefix,
                                cache_key=_cache_key,
                                system_prompt="You are a strict judge. Output JSON only.",
                                max_tokens=64,
                            )
                            # Parse JSON list
                            import json as _json
                            if isinstance(raw, str):
                                s = raw.strip()
                                # strict json first
                                try:
                                    arr = _json.loads(s)
                                except Exception:
                                    # fallback: extract first [...] block
                                    import re as _re
                                    m = _re.search(r'\[[\s\S]*\]', s)
                                    arr = _json.loads(m.group(0)) if m else None
                                if isinstance(arr, list):
                                    judged = []
                                    for v in arr:
                                        if isinstance(v, bool):
                                            judged.append(1.0 if v else 0.0)
                                        else:
                                            try:
                                                fv = float(v)
                                            except Exception:
                                                fv = 0.0
                                            judged.append(1.0 if fv >= 0.5 else 0.0)
                    except Exception:
                        judged = None
                # 3) Apply scores (fallback to per-segment judge if needed)
                need_fallback = (judged is None) or (not isinstance(judged, list)) or (len(judged) != len(segments_meta))
                for idx, seg in enumerate(segments_meta):
                    score = seg.get('forced', None)
                    if score is None:
                        if not need_fallback:
                            score = float(judged[idx])
                        else:
                            try:
                                v = self._evaluate_reasoning_segment(
                                    segment_text=seg.get('text', ''),
                                    documents=documents,
                                    evidences=evidences,
                                    question=question,
                                    ground_truth=ground_truth,
                                    answer_aliases=answer_aliases,
                                    answerable=answerable,
                                )
                                score = float(v) if v is not None else 0.0
                            except Exception:
                                score = 0.0
                    set_span_value(seg['global_start'], seg['global_end'], float(score))
                    step_wise_scores.append(float(score))
        if torch.distributed.get_rank() == 0:
            pass
            char_pos = 0
            for i,(score,ids) in enumerate(zip(sentence_mask,valid_response_ids)):
                token_text = self.input_tokenizer.decode([ids])
                token_len = len(token_text)
                char_pos += token_len
        special_tags = [r'<think>', r'</think>', r'<answer>', r'</answer>']
        for tag_pattern in special_tags:
             for m in re.finditer(tag_pattern, response):
                 set_span_value(m.start(), m.end(), 1.0)
        # Set answer content mask to 2.0
        for m in re.finditer(answer_pattern, response, re.DOTALL):
             set_span_value(m.start(1), m.end(1), 2.0)
        if len(step_wise_scores) > 0:
            try:
                reasoning_score = float(np.mean(step_wise_scores))
            except Exception:
                reasoning_score = float(sum(step_wise_scores) / len(step_wise_scores))
        return reasoning_score, sentence_mask
    def _evaluate_reasoning_segment(self, segment_text, documents, evidences, question=None, ground_truth=None, answer_aliases=None, answerable=True):
        try:
            has_idk = self._check_insufficient_info(segment_text, question=question)
            if has_idk and answerable is False:
                return 1.0
            elif has_idk and answerable is True:
                return -1.0
            import ast
            evd = evidences
            if isinstance(evidences, str):
                try:
                    evd = ast.literal_eval(evidences)
                except Exception:
                    evd = evidences
            def _to_text(x):
                try:
                    return str(x)
                except Exception:
                    return ''
            evd_list = evd if isinstance(evd, (list, tuple)) else [evd]
            evd_texts = [_to_text(x) for x in evd_list if _to_text(x)]
            max_evd = 8
            evd_text = '\n'.join([f"- {t}" for t in evd_texts[:max_evd]]) if evd_texts else ""
            answers_for_context = []
            if ground_truth is not None:
                if isinstance(ground_truth, (list, tuple)):
                    answers_for_context.extend([_to_text(a) for a in ground_truth if _to_text(a)])
                else:
                    answers_for_context.append(_to_text(ground_truth))
            if answer_aliases:
                if isinstance(answer_aliases, (list, tuple)):
                    answers_for_context.extend([_to_text(a) for a in answer_aliases if _to_text(a)])
            answers_for_context = [a for a in answers_for_context if a]
            answers_block = '\n'.join([f"- {a}" for a in answers_for_context]) if answers_for_context else ""
            q_text = (question or '').strip()
            seg_text = (segment_text or '').strip()
            # NOTE: To maximize vLLM prefix/KV-cache reuse across stepwise segments,
            # we structure the prompt as: [static prefix incl. evidences] + [segment-specific suffix].
            prefix_lines = []
            prefix_lines.append("You are a strict reasoning consistency judge. Decide if the reasoning segment is FULLY SUPPORTED by the provided evidences.")
            prefix_lines.append("Rules:")
            prefix_lines.append("1) Output only one digit: 1 if the segment contains meaningful reasoning AND is strictly supported by the evidences; 0 otherwise.")
            prefix_lines.append("2) Give 0 if the segment adds NO new information, is just a plan/re-statement, or lacks specific details (e.g., 'We should review the list...').")
            prefix_lines.append("3) Give 1 ONLY if the segment's key assertion semantically matches or is directly inferred from an evidence.")
            prefix_lines.append("4) Base the decision strictly on the evidences; ignore world knowledge.")
            prefix_lines.append("5) Do not provide explanations.")
            if evd_text:
                prefix_lines.append("")
                prefix_lines.append(f"Evidences:\n{evd_text}")
            prefix_lines.append("")
            prefix_lines.append("Reasoning Segment:")
            prefix = '\n'.join(prefix_lines) + "\n"
            suffix = f"{seg_text}\n\nOutput (only 0 or 1):"
            judge_prompt = prefix + suffix
            from verl.utils.reward_score.answer_postprocessor import get_postprocessor
            post = get_postprocessor()
            if not getattr(post, 'use_judge_api', False):
                return None
            # Provide prefix hint for KV-cache-aware FLOPs accounting (best-effort).
            import hashlib as _hashlib
            _cache_key = "reasoning_seg_prefix:" + _hashlib.sha1(prefix.encode("utf-8", errors="ignore")).hexdigest()
            result_text = post._call_judge_api(judge_prompt, cache_prefix=prefix, cache_key=_cache_key)
            if not isinstance(result_text, str):
                return None
            s = result_text.strip()
            if s in ('0', '1'):
                return 1.0 if s == '1' else 0.0
            import re as _re
            m = _re.match(r'^\s*([01])', s)
            if m:
                return 1.0 if m.group(1) == '1' else 0.0
            return None
        except Exception:
            return None
    def _check_insufficient_info(self, segment_text, question=None):
        seg_text = (segment_text or '').strip()
        if not seg_text:
            return False
        t = seg_text.lower()
        likely_markers = [
            "insufficient",
            "no answer",
            "not enough information",
            "not enough data",
            "cannot determine",
            "can't determine",
            "cannot answer",
            "can't answer",
            "i don't know",
            "i dont know",
            "unknown",
            "no sufficient information",
            "no information",
            "lack of information",
            "lacking information",
        ]
        return any(m in t for m in likely_markers)
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data = data.to(torch.cuda.current_device())
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        answer_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        reasoning_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        sentence_mask_tensor = torch.ones_like(data.batch['responses'], dtype=torch.float32)
        # perform forward computation
        with self.ulysses_sharding_manager:
            # Track judge-server FLOPs (if enabled) during this call
            try:
                from verl.utils.reward_score.answer_postprocessor import get_postprocessor
                post = get_postprocessor()
                _judge_flops_before = float(getattr(post, "_judge_server_flops_total", 0.0))
            except Exception:
                post = None
                _judge_flops_before = 0.0
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
                sequences_str = self.input_tokenizer.decode(sequences)
                sequences_input = self.input_tokenizer.decode(valid_response_ids)
                documents = data_item.non_tensor_batch['documents'] # [[key, [list]]]
                ground_truth = data_item.non_tensor_batch['answer'] # str
                data_source = data_item.non_tensor_batch['data_source']
                strategy = os.environ.get('STRATEGY', None)
                compute_score_fn = self._select_rm_score_fn(data_source)
                # extract answerable strictly from extra_info
                extra_info = data_item.non_tensor_batch.get('extra_info', {})
                if isinstance(extra_info, str):
                    try:
                        import json as _json
                        extra_info = _json.loads(extra_info)
                    except Exception:
                        extra_info = {}
                if not isinstance(extra_info, dict):
                    extra_info = {}
                raw_flag = extra_info.get('answerable', True)
                if isinstance(raw_flag, str):
                    rf = raw_flag.strip().lower()
                    answerable_flag = True if rf == 'true' else False if rf == 'false' else True
                elif isinstance(raw_flag, bool):
                    answerable_flag = raw_flag
                else:
                    answerable_flag = True
                # Extract answer_aliases for data sources that support it
                answer_aliases = extra_info.get('answer_aliases', []) if isinstance(extra_info, dict) else []
                # Call compute_score_fn with or without answer_aliases based on data source
                if data_source in ['hotpot', '2wikimultihop', 'musique']:
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
                        answerable=answerable_flag,
                    )
                # Note: base_reward is not used in training, only stored for validation
                evidences = data_item.non_tensor_batch['evidences']
                # Default total_score calculation (can be overridden by strategies)
                resp_len_int = int(valid_response_length.item() if hasattr(valid_response_length, "item") else int(valid_response_length))
                if resp_len_int > 500:
                    excess_tokens = resp_len_int - 500
                    excess_hundreds = (excess_tokens + 99) // 100
                    length_penalty = 0.2 * excess_hundreds
                else:
                    length_penalty = 0.0
                total_score = float(answer_score) - length_penalty
                if strategy == 'grpo':
                    reasoning_score = 0
                    sentence_mask_tensor[i, :] = 0.0
                elif strategy == 'knowrl':
                    # KnowRL: R_total = r_format + r_correct + r_fact
                    # 1. r_format: +1 if correct, -1 if incorrect
                    if format_score == -2:
                        r_format = -1.0
                    else:
                        r_format = 1.0
                    # 2. r_fact: (true count / total count) -> 0~1
                    # validate_model_reasoning_documents_only returns mean of scores.
                    # We set positive=1.0, negative=0.0 so mean is exactly true/total.
                    # If total=0, it returns 0.0.
                    reasoning_score, sentence_mask = self.validate_model_reasoning_documents_only(
                        documents, 
                        sequences_input,
                        valid_response_ids=valid_response_ids,
                        positive_score=1.0, 
                        negative_score=0.0
                    )
                    r_fact = reasoning_score
                    # Also populate sentence_mask_tensor for potential logging/analysis
                    mask_length = min(len(valid_response_ids), len(sentence_mask))
                    sentence_mask_tensor[i, :mask_length] = torch.tensor(sentence_mask[:mask_length], dtype=torch.float32)
                    # 3. r_correct: +2 correct, -1 incorrect, +1 IDK
                    if answer_score == 1.0:
                        r_correct = 2.0
                    else:
                        # Check for IDK in the answer part
                        try:
                            question_text = data_item.non_tensor_batch.get('question', None)
                        except:
                            question_text = None
                        # Extract answer text to check for IDK
                        ans_str = self.extract_solution(sequences_input)
                        if self._check_insufficient_info(ans_str, question=question_text):
                            r_correct = 1.0
                        else:
                            r_correct = -1.0
                    # Override total_score
                    total_score = r_format + r_correct + r_fact
                else:
                    # Initialize current row to 0 first (since default init was ones)
                    sentence_mask_tensor[i, :] = 0.0
                    if format_score == -2:
                        reasoning_score = 0
                        sentence_mask_tensor[i, :] = 0.0
                        # Removed verbose print to reduce log clutter
                        # print(f"\n[Reasoning Validation] Skipped due to format errors (Reasoning score: {reasoning_score})")
                    else:
                        # CRITICAL: Pass valid_response_ids directly to ensure alignment
                        # try to get question if available in batch (optional)
                        try:
                            question_text = data_item.non_tensor_batch['question']
                        except Exception:
                            question_text = None
                        if strategy == 'fspo':
                            evidences = documents
                        reasoning_score, sentence_mask = self.validate_model_reasoning_stepwise(
                            sequences_input,
                            documents,
                            evidences,
                            valid_response_ids=valid_response_ids,
                            question=question_text,
                            ground_truth=ground_truth,
                            answer_aliases=answer_aliases,
                            answerable=answerable_flag,
                        )
                        # sentence_mask length MUST match valid_response_ids length
                        mask_length = min(len(valid_response_ids), len(sentence_mask))
                        sentence_mask_tensor[i, :mask_length] = torch.tensor(sentence_mask[:mask_length], dtype=torch.float32)
                # Removed verbose prints to reduce log clutter
                # print("\n" + "-" * 80)
                # print(f" Final Score ".center(80, '-'))
                # print(f"  Format score: {format_score}")
                # print(f"  Answer score: {answer_score}")
                # print(f"  Reasoning score: {reasoning_score}")
                # Logic moved to start of loop to allow override
                # try:
                #     resp_len_int = int(valid_response_length.item() if hasattr(valid_response_length, "item") else int(valid_response_length))
                # except Exception:
                #     # Fallback: best-effort conversion
                #     resp_len_int = int(valid_response_length) if not isinstance(valid_response_length, torch.Tensor) else int(valid_response_length.detach().cpu().item())
                # if resp_len_int > 500:
                #     excess_tokens = resp_len_int - 500
                #     excess_hundreds = (excess_tokens + 99) // 100
                #     length_penalty = 0.1 * excess_hundreds
                # else:
                #     length_penalty = 0.0
                # total_score = float(answer_score) - length_penalty
                reward_tensor[i, valid_response_length - 1] = total_score
                format_reward_tensor[i] = format_score
                answer_reward_tensor[i] = answer_score
                reasoning_reward_tensor[i] = reasoning_score
            for sample_idx in range(min(3, sentence_mask_tensor.shape[0])):
                sample_mask = sentence_mask_tensor[sample_idx]
                nonzero_count = (sample_mask != 0).sum().item()
                unique_values = torch.unique(sample_mask).cpu().tolist()
            output = DataProto.from_dict(tensors={'reward_scores': reward_tensor,
                                                  'format_scores': format_reward_tensor,
                                                  'answer_scores': answer_reward_tensor,
                                                  'reasoning_scores': reasoning_reward_tensor,
                                                  'sentence_mask': sentence_mask_tensor})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
        output = output.to('cpu')
        # Collect judge-server FLOPs accrued in this compute_rm_score call
        try:
            judge_flops = float(post.consume_judge_server_flops()) if post is not None else 0.0
        except Exception:
            judge_flops = 0.0
        output.meta_info = dict(output.meta_info or {})
        output.meta_info['flops_step'] = judge_flops
        return output