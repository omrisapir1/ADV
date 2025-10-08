from typing import List, Dict, Any, Optional

from vllm import LLM, SamplingParams
import torch


class VLLMEngineWrapper:
    def __init__(self, model_name: str, gpu_id: int, vllm_config: Optional[Dict[str, Any]] = None):
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Extract vllm configuration parameters
        vllm_config = vllm_config or {}
        gpu_memory_utilization = vllm_config.get("gpu_memory_utilization")
        max_num_seqs = vllm_config.get("max_num_seqs")
        max_num_batched_tokens = vllm_config.get("max_num_batched_tokens")

        self.backend = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=False,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens
        )


    def generate_candidates(self, prompts: List[str], n_samples: int, **gen_cfg) -> List[List[str]]:
        temperature = float(gen_cfg.get("temperature", 0.7))
        top_p = float(gen_cfg.get("top_p", 0.9))
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 64))
        repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=n_samples,
            repetition_penalty=repetition_penalty
        )
        outputs = self.backend.generate(prompts, sampling_params)
        all_candidates: List[List[str]] = []
        for out in outputs:
            cand = [seg.text for seg in out.outputs]
            all_candidates.append(cand)
        return all_candidates


def build_vllm_engine(model_name: str, gpu_id: int, vllm_config: Optional[Dict[str, Any]] = None) -> VLLMEngineWrapper:
    return VLLMEngineWrapper(model_name=model_name, gpu_id=gpu_id, vllm_config=vllm_config)
