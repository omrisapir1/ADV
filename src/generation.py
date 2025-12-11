from __future__ import annotations

import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple
import re
import time

import requests
from openai import AsyncOpenAI


class EngineCircuitBreaker(Exception):
    """Raised when engine encounters too many consecutive empty generations."""
    pass

class AsyncSGLangEngineWrapper:
    """Simplified SGLang engine wrapper using generation config (with concurrency & resilience)."""
    def __init__(self, model_name: str, sglang_config: Optional[Dict[str, Any]] = None):
        sglang_config = sglang_config or {}
        base_url = "http://localhost:30000/v1"
        api_key  = "EMPTY"
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

        max_conc = sglang_config.get("max_concurrency")
        if not isinstance(max_conc, int) or max_conc <= 0:
            raise ValueError("max_concurrency must be a positive int in generation config")
        self._semaphore = asyncio.Semaphore(max_conc)

        self.per_request_timeout = float(sglang_config.get("per_request_timeout", 120.0))
        # phase2_batch_limit not used anymore in single-phase mode but keep default for config stability
        self.phase2_batch_limit = int(sglang_config.get("phase2_batch_limit", max_conc))
        if self.phase2_batch_limit <= 0:
            self.phase2_batch_limit = max_conc
        self.circuit_breaker_failures = int(sglang_config.get("circuit_breaker_failures", 5))
        self._consecutive_failures = 0

        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "timeouts": 0,
            "errors": 0,
            "total_time": 0.0,
            "in_flight": 0,
            # phase2_batches retained for backward compat but no longer incremented
            "phase2_batches": 0,
            "circuit_breaker_trips": 0,
        }

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self.metrics)

    def hot_swap(self, tmp_weights_path: str):
        url = "http://localhost:30000/update_weights_from_disk"
        data = {"model_path": tmp_weights_path}
        try:
            requests.post(url, json=data)
        except Exception:
            pass

    async def _completion_call(self, **kwargs):
        async with self._semaphore:
            self.metrics["in_flight"] += 1
            start = time.monotonic()
            task = asyncio.create_task(self.client.completions.create(**kwargs))
            try:
                resp = await asyncio.wait_for(task, timeout=self.per_request_timeout)
                self.metrics["total_requests"] += 1
                return resp
            except asyncio.TimeoutError:
                self.metrics["timeouts"] += 1
                class Dummy: choices = []
                return Dummy()
            except Exception:
                self.metrics["errors"] += 1
                class Dummy: choices = []
                return Dummy()
            finally:
                self.metrics["total_time"] += (time.monotonic() - start)
                self.metrics["in_flight"] -= 1

    async def _sample_for_one_prompt(
        self,
        prompt: str,
        *,
        n_samples: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_new_tokens: int,
        stop: Optional[List[str]],
    ) -> List[str]:
        """Single-phase sampling for one prompt. Returns list of generated strings.
        If generation fails, returns n_samples empty strings.
        """
        payload_extra = {"top_k": top_k, "repetition_penalty": repetition_penalty}
        resp = await self._completion_call(
            model=self.model_name,
            prompt=prompt,
            n=n_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop if stop else None,
            extra_body=payload_extra,
        )
        choices = getattr(resp, "choices", []) or []
        if not choices:
            return [""] * n_samples
        out: List[str] = []
        for ch in choices:
            text = getattr(ch, "text", None) or ""
            out.append(text)
        # Pad if fewer choices than requested
        if len(out) < n_samples:
            out.extend([""] * (n_samples - len(out)))
        return out

    async def generate_candidates(
        self,
        prompts: List[str],
        n_samples: int,
        **gen_cfg: Any,
    ) -> List[List[str]]:
        """Generate candidates for each prompt using single-phase sampling.
        Preserves concurrency, per-request timeout, metrics, and circuit breaker behavior.
        """
        temperature = gen_cfg.get("temperature")
        top_p = gen_cfg.get("top_p")
        top_k = gen_cfg.get("top_k")
        repetition_penalty = gen_cfg.get("repetition_penalty")
        max_new_tokens = gen_cfg.get("max_new_tokens")
        TIMEOUT_SEC = gen_cfg.get("timeout", 280)

        tasks = []
        for p in prompts:
            coro = self._sample_for_one_prompt(
                p,
                n_samples=n_samples,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
            async def run_with_timeout(coro=coro, prompt=p):
                try:
                    return await asyncio.wait_for(coro, timeout=TIMEOUT_SEC)
                except asyncio.TimeoutError:
                    return [""] * n_samples
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return [""] * n_samples
            tasks.append(asyncio.create_task(run_with_timeout()))

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise

        normalized: List[List[str]] = []
        empty_all = True
        for r in results:
            if isinstance(r, Exception):
                normalized.append([""] * n_samples)
            else:
                normalized.append(r)
                # Check if this prompt returned any non-empty strings
                if any(s for s in r):
                    empty_all = False
        if empty_all:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0
        if self._consecutive_failures >= self.circuit_breaker_failures:
            self.metrics["circuit_breaker_trips"] += 1
            self._consecutive_failures = 0
            raise EngineCircuitBreaker("Too many consecutive empty generations; circuit breaker tripped")
        return normalized


def build_sglang_engine(model_name: str, sglang_config: Optional[Dict[str, Any]] = None) -> AsyncSGLangEngineWrapper:
    return AsyncSGLangEngineWrapper(model_name=model_name, sglang_config=sglang_config)