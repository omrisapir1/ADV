from __future__ import annotations

import asyncio, random, os, re, json
from typing import List, Dict, Any, Optional, Tuple, Callable, Awaitable
import requests
from openai import AsyncOpenAI
from openai._exceptions import APIStatusError, RateLimitError, APITimeoutError
import httpx  # comes with the OpenAI client

THINK_STOP = "</think>"
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

def _exp_backoff_sleep(attempt: int, base: float, jitter: float) -> float:
    # backoff = base * 2^(attempt-1)  with full jitter in [0, jitter]
    return base * (2 ** (attempt - 1)) + random.random() * jitter

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
        self.req_timeout = sglang_config.get("request_timeout_s", 60.0)
        self.max_retries = int(sglang_config.get("max_retries", 5))
        self.backoff_base = float(sglang_config.get("backoff_base_s", 0.5))
        self.backoff_jitter = float(sglang_config.get("backoff_jitter_s", 0.5))

    def hot_swap(self, tmp_weights_path: str):
        """Retry POST to SGLang update endpoint (best-effort)."""
        url = self.client.base_url.replace("/v1", "") + "/update_weights_from_disk"
        payload = {"model_path": tmp_weights_path}

        # Small manual retry for the control-plane POST
        last_err = None
        for attempt in range(1, 4):
            try:
                resp = requests.post(url, json=payload, timeout=self.req_timeout)
                if 200 <= resp.status_code < 300:
                    return
                last_err = RuntimeError(f"hotswap http {resp.status_code}: {resp.text[:200]}")
            except requests.RequestException as e:
                last_err = e
            # backoff
            sleep_s = _exp_backoff_sleep(attempt, 0.25, 0.25)
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(sleep_s))
        raise last_err or RuntimeError("hotswap failed")

    async def _retry(
        self,
        coro_factory: Callable[[], Awaitable[Any]],
        label: str = "api_call",
    ) -> Any:
        """Retry helper for transient errors with exponential backoff + jitter."""
        attempt = 1
        while True:
            try:
                return await coro_factory()
            except (APITimeoutError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                # Always retry on timeouts up to max_retries
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(_exp_backoff_sleep(attempt, self.backoff_base, self.backoff_jitter))
                attempt += 1
            except RateLimitError as e:
                # Retry on rate limits
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(_exp_backoff_sleep(attempt, self.backoff_base, self.backoff_jitter))
                attempt += 1
            except APIStatusError as e:
                # Retry on selected 5xx or 429
                status = getattr(e, "status_code", None)
                if status in TRANSIENT_STATUS and attempt < self.max_retries:
                    await asyncio.sleep(_exp_backoff_sleep(attempt, self.backoff_base, self.backoff_jitter))
                    attempt += 1
                else:
                    raise
            except httpx.HTTPError as e:
                # Network glitches
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(_exp_backoff_sleep(attempt, self.backoff_base, self.backoff_jitter))
                attempt += 1

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
        async with self._semaphore:
            # PHASE 1: think
            resp1 = await self._retry(
                lambda: self.client.completions.create(
                    model=self.model_name,
                    prompt=base_prompt,
                    n=n_samples,
                    temperature=think_temperature,
                    top_p=think_top_p,
                    max_tokens=think_max_new_tokens,
                    stop=[THINK_STOP],
                    extra_body=payload_extra_1,
                ),
                label="phase1_think",
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
            async with self._semaphore:
                return await self._retry(
                    lambda: self.client.completions.create(
                        model=self.model_name,
                        prompt=ctx,
                        n=1,
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=answer_max_new_tokens,
                        stop=answer_stop if answer_stop else None,
                        extra_body=payload_extra_2,
                    ),
                    label="phase2_answer",
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

        coros = [
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
            ) for p in prompts
        ]
        # Protect the whole batch from failing if one prompt errors
        results = await asyncio.gather(*[asyncio.create_task(c) for c in coros], return_exceptions=True)
        # Normalize exceptions -> empty results for that prompt
        out: List[List[tuple[str, int]]] = []
        for r in results:
            if isinstance(r, Exception):
                out.append([])
            else:
                out.append(r)
        return out


def build_sglang_engine(model_name: str, sglang_config: Optional[Dict[str, Any]] = None) -> AsyncSGLangEngineWrapper:
    return AsyncSGLangEngineWrapper(model_name=model_name, sglang_config=sglang_config)
