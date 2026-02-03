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
import torch
from transformers import PretrainedConfig, Qwen2Config, LlamaConfig
VALID_CONFIG_TYPE = (Qwen2Config, LlamaConfig)
def get_device_flops(unit="T"):
    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number
    device_name = torch.cuda.get_device_name()
    flops = float("inf")  # INF flops for unkown gpu type
    if "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12
    flops_unit = unit_convert(flops, unit)
    return flops_unit
class FlopsCounter:
    """
    Used to count mfu during training loop
    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)
    """
    def __init__(self, config: PretrainedConfig):
        if not isinstance(config, VALID_CONFIG_TYPE):
            print(f"Only support config type of {VALID_CONFIG_TYPE}, but got {type(config)}. "
                  f"MFU will always be zero.")
        self.estimate_func = {"qwen2": self._estimate_qwen2_flops, 'llama': self._estimate_qwen2_flops}
        self.config = config
    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time):
        return 0
    def _total_flops_unknown(self, tokens_sum, batch_seqlens):
        return 0.0
    def _total_flops_qwen2(self, tokens_sum, batch_seqlens):
        """
        Return total FLOPs (not FLOPs/s) for one forward+backward pass on the given batch.
        Notes:
        - This uses the same analytic formula as `_estimate_qwen2_flops`, but without dividing by time.
        - This is intended for *training FLOPs consumption* accounting.
        """
        assert isinstance(self.config, (Qwen2Config, LlamaConfig))
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size
        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        # non-attn per layer parm
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum
        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
        return float(dense_N_flops + attn_qkv_flops)
    def _total_flops_qwen2_forward(self, tokens_sum, batch_seqlens):
        """
        Forward-only FLOPs estimate for the same analytic model.
        We conservatively approximate forward-only as 1/3 of (forward+backward) in `_total_flops_qwen2`.
        """
        return float(self._total_flops_qwen2(tokens_sum, batch_seqlens) / 3.0)
    def _total_flops_qwen2_forward_linear(self, tokens_sum):
        """
        Forward-only FLOPs (linear-only; ignore quadratic attention term).
        This is useful for inference services (e.g., server-side LLM judge / vLLM) to avoid overcounting KV-cache behavior.
        """
        assert isinstance(self.config, (Qwen2Config, LlamaConfig))
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size
        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # Forward-only dense FLOPs: 2 * params * tokens (matches the convention used by estimate_qwen2_flops via 6x for train)
        return float(2 * dense_N * tokens_sum)
    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        assert isinstance(self.config, (Qwen2Config, LlamaConfig))
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size
        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        # non-attn per layer parm
        # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum
        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved
    def estimate_flops(self, batch_seqlens, delta_time):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.
        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.
        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
    def estimate_total_flops(self, batch_seqlens) -> float:
        """
        Estimate total FLOPs (not FLOPs/s) for one forward+backward pass on the given batch.
        This is designed for *FLOPs consumption accounting* over the full training run.
        Unsupported model types return 0.0 (conservative; will undercount rather than overcount).
        """
        tokens_sum = sum(batch_seqlens)
        total_func = {
            "qwen2": self._total_flops_qwen2,
            "llama": self._total_flops_qwen2,
        }.get(self.config.model_type, self._total_flops_unknown)
        return float(total_func(tokens_sum, batch_seqlens))
    def estimate_total_flops_forward(self, batch_seqlens) -> float:
        """
        Estimate total forward-only FLOPs for a full-sequence forward (no backward).
        """
        tokens_sum = sum(batch_seqlens)
        total_func = {
            "qwen2": self._total_flops_qwen2_forward,
            "llama": self._total_flops_qwen2_forward,
        }.get(self.config.model_type, lambda _ts, _bs: 0.0)
        return float(total_func(tokens_sum, batch_seqlens))
    def estimate_total_flops_forward_linear(self, tokens_sum: int) -> float:
        """
        Estimate forward-only FLOPs with a linear-only model (ignore quadratic attention).
        Intended for server-side inference accounting to avoid overcounting.
        """
        if tokens_sum <= 0:
            return 0.0
        if self.config.model_type not in ("qwen2", "llama"):
            return 0.0
        return float(self._total_flops_qwen2_forward_linear(tokens_sum))