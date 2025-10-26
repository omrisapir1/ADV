from __future__ import annotations

import asyncio
import random
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
            """Greedy second-phase completion with overall 2 min deadline and retries."""
            max_total_seconds = 120.0
            per_attempt_timeout = 30.0  # cap per attempt
            max_attempts = 5
            base_backoff = 5.0
            deadline = asyncio.get_event_loop().time() + max_total_seconds
            attempt = 1
            while attempt <= max_attempts:
                now = asyncio.get_event_loop().time()
                remaining = deadline - now
                if remaining <= 0:
                    print(f"[greedy] Global deadline exceeded for ctx (attempt {attempt}). Skipping.")
                    class _Dummy:
                        choices: list = []
                    return _Dummy()
                attempt_timeout = min(per_attempt_timeout, remaining)
                try:
                    async with self._semaphore:
                        resp = await asyncio.wait_for(
                            self.client.completions.create(
                                model=self.model_name,
                                prompt=ctx,
                                n=1,
                                temperature=0.0,
                                top_p=1.0,
                                max_tokens=answer_max_new_tokens,
                                stop=answer_stop if answer_stop else None,
                                extra_body=payload_extra_2,
                            ),
                            timeout=attempt_timeout,
                        )
                    return resp
                except asyncio.TimeoutError:
                    print(f"[greedy] Attempt {attempt} timed out after {attempt_timeout:.1f}s")
                except (RateLimitError, APIStatusError) as e:
                    print(f"[greedy] Attempt {attempt} API error: {type(e).__name__}: {e}")
                except asyncio.CancelledError:
                    # Propagate cancellations
                    raise
                except Exception as e:
                    print(f"[greedy] Attempt {attempt} unexpected error: {e}")
                # Prepare next attempt
                attempt += 1
                if attempt > max_attempts:
                    break
                backoff = min(base_backoff * (2 ** (attempt - 2)) + random.uniform(0, 1), remaining)
                # If backoff exceeds remaining time, abort early
                if asyncio.get_event_loop().time() + backoff > deadline:
                    print("[greedy] Not enough time left for another attempt; aborting.")
                    break
                await asyncio.sleep(backoff)
            print(f"[greedy] Exhausted attempts for ctx; returning empty dummy response.")
            class _Dummy:
                choices: list = []
            return _Dummy()
        tasks = [asyncio.create_task(_greedy(ctx)) for _, _, ctx in phase2_items]
        try:
            resp2_list = await asyncio.wait_for(asyncio.gather(*tasks), timeout=130.0)  # 2min + buffer
        except asyncio.TimeoutError:
            print("[phase2] Global 2-min timeout reached; cancelling unfinished tasks.")
            resp2_list = []
            for t in tasks:
                if not t.done():
                    t.cancel()
            for t in tasks:
                try:
                    r = await t
                except asyncio.CancelledError:
                    class _Dummy: choices: list = []
                    r = _Dummy()
                except Exception as e:
                    print(f"[phase2] Task error after cancel: {e}")
                    class _Dummy: choices: list = []
                    r = _Dummy()
                resp2_list.append(r)
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
