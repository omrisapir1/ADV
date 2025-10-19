from __future__ import annotations

import asyncio
import random
import os
from typing import List, Dict, Any, Optional, Tuple
import requests
from openai import AsyncOpenAI
from openai._exceptions import APIStatusError, RateLimitError
import re

THINK_STOP = "</think>"

class AsyncSGLangEngineWrapper:
    """Simplified SGLang engine wrapper using generation config (no default fallbacks)."""
    def __init__(self, model_name: str, sglang_config: Optional[Dict[str, Any]] = None):
        sglang_config = sglang_config or {}
        # Base URL / API key fixed unless provided elsewhere; user manages server separately.
        base_url = "http://localhost:30000/v1"
        api_key  = "EMPTY"
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        # Concurrency from config; will raise if missing.
        self._semaphore = asyncio.Semaphore(sglang_config.get("max_concurrency"))

    def hot_swap(self, tmp_weights_path: str):
        url = "http://localhost:30000/update_weights_from_disk"
        data = {"model_path": tmp_weights_path}
        requests.post(url, json=data)

    async def _two_phase_for_one_prompt(
        self,
        base_prompt: str,
        *,
        n_samples: int,
        think_temperature: float,
        think_top_p: float,
        think_max_new_tokens: int,
        think_top_k: int,
        think_repetition_penalty: float,
        answer_max_new_tokens: int,
        answer_stop: List[str],
    ) -> List[tuple[str, int]]:
        """Two-phase generation for a single prompt."""
        payload_extra_1 = {"top_k": think_top_k, "repetition_penalty": think_repetition_penalty}
        resp1 = await self.client.completions.create(
            model=self.model_name,
            prompt=base_prompt,
            n=n_samples,
            temperature=think_temperature,
            top_p=think_top_p,
            max_tokens=think_max_new_tokens,
            stop=[THINK_STOP],
            extra_body=payload_extra_1,
        )
        results: List[tuple[str, int]] = [("", 0)] * len(resp1.choices)
        phase2_items: List[Tuple[int, str, str]] = []
        for idx, choice in enumerate(resp1.choices):
            think_piece = (choice.text or "")
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason != "stop" or re.findall(r"\\boxed\s*\{(.*?)\}", think_piece or "", flags=re.DOTALL):
                results[idx] = (think_piece, 0)
                continue
            think_clean = think_piece.split(THINK_STOP, 1)[0] if THINK_STOP in think_piece else think_piece
            context = base_prompt + think_clean + THINK_STOP
            phase2_items.append((idx, think_clean, context))
        if not phase2_items:
            return results
        payload_extra_2 = {"top_k": 0, "repetition_penalty": 1.0}
        async def _greedy(ctx: str):
            try:
                return await self.client.completions.create(
                    model=self.model_name,
                    prompt=ctx,
                    n=1,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=answer_max_new_tokens,
                    stop=answer_stop if answer_stop else None,
                    extra_body=payload_extra_2,
                )
            except :
                print('Proba;y reached time out will re-try in 10 seconds')
                await asyncio.sleep(10)
                return await self.client.completions.create(
                    model=self.model_name,
                    prompt=ctx,
                    n=1,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=answer_max_new_tokens,
                    stop=answer_stop if answer_stop else None,
                )
        tasks = [asyncio.create_task(_greedy(ctx)) for _, _, ctx in phase2_items]
        resp2_list = await asyncio.gather(*tasks)
        for (idx, think_clean, _), resp2 in zip(phase2_items, resp2_list):
            answer_text = (resp2.choices[0].text or "") if resp2.choices else ""
            full_text = think_clean + THINK_STOP + answer_text
            results[idx] = (full_text, 1)
        return results

    async def generate_candidates(
        self,
        prompts: List[str],
        n_samples: int,
        **gen_cfg: Any,
    ) -> List[List[tuple[str, int]]]:
        """Generate candidates for each prompt using two-phase method with config values."""
        think_temperature = gen_cfg.get("think_temperature")
        think_top_p = gen_cfg.get("think_top_p")
        think_top_k = gen_cfg.get("think_top_k")
        think_repetition_penalty = gen_cfg.get("think_repetition_penalty")
        think_max_new_tokens = gen_cfg.get("think_max_new_tokens")
        answer_max_new_tokens = gen_cfg.get("answer_max_new_tokens")
        answer_stop = gen_cfg.get("answer_stop")
        tasks = [
            asyncio.create_task(
                self._two_phase_for_one_prompt(
                    p,
                    n_samples=n_samples,
                    think_temperature=think_temperature,
                    think_top_p=think_top_p,
                    think_max_new_tokens=think_max_new_tokens,
                    think_top_k=think_top_k,
                    think_repetition_penalty=think_repetition_penalty,
                    answer_max_new_tokens=answer_max_new_tokens,
                    answer_stop=answer_stop,
                )
            ) for p in prompts
        ]
        return await asyncio.gather(*tasks)


def build_sglang_engine(model_name: str, sglang_config: Optional[Dict[str, Any]] = None) -> AsyncSGLangEngineWrapper:
    return AsyncSGLangEngineWrapper(model_name=model_name, sglang_config=sglang_config)
