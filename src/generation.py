from typing import List, Dict, Any, Optional
import os

from openai import OpenAI


class SGLangEngineWrapper:
    def __init__(self, model_name: str, gpu_id: int = 0, sglang_config: Optional[Dict[str, Any]] = None):
        _ = gpu_id
        sglang_config = sglang_config or {}
        base_url = sglang_config.get("base_url") or os.environ.get("SGLANG_BASE_URL", "http://localhost:30000/v1")
        api_key  = sglang_config.get("api_key")  or os.environ.get("SGLANG_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def generate_candidates(self, prompts: List[str], n_samples: int, **gen_cfg) -> List[List[str]]:
        """RAW prompt path — no chat template applied."""
        temperature = float(gen_cfg.get("temperature"))
        top_p = float(gen_cfg.get("top_p"))
        max_new_tokens = int(gen_cfg.get("max_new_tokens"))
        repetition_penalty = float(gen_cfg.get("repetition_penalty"))
        top_k = int(gen_cfg.get("top_k", 0))

        all_candidates: List[List[str]] = []
        for prompt in prompts:
            resp = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,           # <- raw, already-crafted prompt
                n=n_samples,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                extra_body={
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    # add any other sglang knobs here
                },
            )
            cand = [c.text or "" for c in resp.choices]
            all_candidates.append(cand)
        return all_candidates

def build_sglang_engine(model_name: str, gpu_id: int, sglang_config: Optional[Dict[str, Any]] = None) -> SGLangEngineWrapper:
    return SGLangEngineWrapper(model_name=model_name, gpu_id=gpu_id, sglang_config=sglang_config)
