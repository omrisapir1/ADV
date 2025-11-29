from __future__ import annotations

import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple
import re
import time
import math

import requests
from openai import AsyncOpenAI


THINK_STOP = "</think>"

class EngineCircuitBreaker(Exception):
    """Raised when engine encounters too many consecutive empty generations."""
    pass

class AsyncSGLangEngineWrapper:
    """Simplified SGLang engine wrapper using generation config (with concurrency & resilience)."""
    def __init__(self, model_name: str, sglang_config: Optional[Dict[str, Any]] = None):
        sglang_config = sglang_config or {}
        base_url = "http://localhost:30000/v1"  # use root; client adds /v1
        api_key  = "EMPTY"
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

        max_conc = sglang_config.get("max_concurrency")
        if not isinstance(max_conc, int) or max_conc <= 0:
            raise ValueError("max_concurrency must be a positive int in generation config")
        self._semaphore = asyncio.Semaphore(max_conc)

        self.per_request_timeout = float(sglang_config.get("per_request_timeout", 120.0))
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
            # Build chat messages from prompt
            prompt_text = kwargs.pop("prompt", "")
            messages = kwargs.pop("messages", None)
            if messages is None:
                messages = [{"role": "user", "content": prompt_text}]
            # Map common args
            payload = dict(
                model=self.model_name,
                messages=messages,
                n=kwargs.pop("n", 1),
                temperature=kwargs.pop("temperature", 0.0),
                top_p=kwargs.pop("top_p", 1.0),
                max_tokens=kwargs.pop("max_tokens", None),
                stop=kwargs.pop("stop", None),
                logprobs=kwargs.pop("logprobs", False),
                top_logprobs=kwargs.pop("top_logprobs", None),
            )
            extra_body = kwargs.pop("extra_body", None)
            if extra_body:
                payload["extra_body"] = extra_body
            task = asyncio.create_task(self.client.chat.completions.create(**payload))
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

    # Entropy helper based on top_logprobs per token
    @staticmethod
    def _entropy_from_top_logprobs(token_info) -> Optional[float]:
        """
        Compute entropy H = -Σ p_i log p_i from top_logprobs for a single position.
        Uses natural log; result is in 'nats'.
        Returns None if probabilities cannot be computed.
        """
        # token_info.top_logprobs expected as a list of objects each with .logprob
        top = getattr(token_info, "top_logprobs", None)
        if not top:
            return None
        probs = []
        for t in top:
            lp = getattr(t, "logprob", None)
            if lp is None:
                continue
            probs.append(math.exp(lp))
        Z = sum(probs)
        if Z <= 0.0:
            return None
        probs = [p / Z for p in probs]
        return -sum(p * math.log(p) for p in probs if p > 0.0)

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
    ) -> List[tuple[str, int, float]]:
        """Two-phase generation for a single prompt with bounded concurrency & cancellation safety.
        Returns a list of (full_text, phase_flag, avg_entropy) where phase_flag: 0=think-only, 1=think+answer.
        avg_entropy computed from phase-1 logprobs for the think segment.
        """
        payload_extra_1 = {"top_k": think_top_k, "repetition_penalty": think_repetition_penalty}
        # Phase 1
        resp1 = await self._completion_call(
            prompt=base_prompt,
            n=n_samples,
            temperature=think_temperature,
            top_p=think_top_p,
            max_tokens=think_max_new_tokens,
            stop=[THINK_STOP],
            extra_body=payload_extra_1,
            logprobs=True,
            top_logprobs=20,
        )
        results: List[tuple[str, int, float]] = [("", 0, float("nan"))] * (len(resp1.choices) if getattr(resp1, "choices", None) else n_samples)
        phase2_items: List[Tuple[int, str, str, Optional[float]]] = []
        for idx, choice in enumerate(getattr(resp1, "choices", [])):
            think_piece = (getattr(choice, "text", "") or "")
            finish_reason = getattr(choice, "finish_reason", None)
            # compute avg entropy from logprobs if present
            avg_entropy: Optional[float] = None
            logprobs_obj = getattr(choice, "logprobs", None)
            content_tokens = getattr(logprobs_obj, "content", None) if logprobs_obj is not None else None
            if content_tokens:
                entropies: List[float] = []
                for token_info in content_tokens:
                    h = self._entropy_from_top_logprobs(token_info)
                    if h is not None:
                        entropies.append(h)
                if entropies:
                    avg_entropy = sum(entropies) / len(entropies)
            if finish_reason != "stop" or re.findall(r"\\boxed\s*{(.*?)}", think_piece or "", flags=re.DOTALL):
                results[idx] = (think_piece, 0, avg_entropy if avg_entropy is not None else float("nan"))
                continue
            think_clean = think_piece.split(THINK_STOP, 1)[0] if THINK_STOP in think_piece else think_piece
            context = base_prompt + think_clean + THINK_STOP
            phase2_items.append((idx, think_clean, context, avg_entropy))
        if not phase2_items:
            return results

        payload_extra_2 = {"top_k": 0, "repetition_penalty": 1.0}

        async def _greedy(ctx: str):
            return await self._completion_call(
                prompt=ctx,
                n=1,
                temperature=0.0,
                top_p=1.0,
                max_tokens=answer_max_new_tokens,
                stop=answer_stop if answer_stop else None,
                extra_body=payload_extra_2,
            )

        # Phase 2 batched
        try:
            for start_idx in range(0, len(phase2_items), self.phase2_batch_limit):
                batch = phase2_items[start_idx:start_idx + self.phase2_batch_limit]
                tasks = [asyncio.create_task(_greedy(ctx)) for _, _, ctx, _ in batch]
                self.metrics["phase2_batches"] += 1
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                for (idx, think_clean, _, avg_entropy), resp2 in zip(batch, gathered):
                    if isinstance(resp2, Exception) or not getattr(resp2, "choices", None):
                        # fallback to think only
                        full_text = think_clean + THINK_STOP
                        results[idx] = (full_text, 0, avg_entropy if avg_entropy is not None else float("nan"))
                        continue
                    answer_text = (resp2.choices[0].text or "") if resp2.choices else ""
                    full_text = think_clean + THINK_STOP + answer_text
                    results[idx] = (full_text, 1, avg_entropy if avg_entropy is not None else float("nan"))
                await asyncio.sleep(random.uniform(0.005, 0.02))  # jitter between batches
        except asyncio.CancelledError:
            # Cancel outstanding tasks if any - tasks already awaited inside loop; just propagate
            raise
        except Exception:
            # In case of unexpected exception, keep existing partial results (think only)
            pass
        return results

    async def generate_candidates(
        self,
        prompts: List[str],
        n_samples: int,
        **gen_cfg: Any,
    ) -> List[List[tuple[str, int, float]]]:
        """Generate candidates for each prompt using two-phase method with config values.
        Implements circuit breaker on repeated empty generations.
        Returns per-prompt list of (full_text, phase_flag, avg_entropy).
        """
        think_temperature = gen_cfg.get("think_temperature")
        think_top_p = gen_cfg.get("think_top_p")
        think_top_k = gen_cfg.get("think_top_k")
        think_repetition_penalty = gen_cfg.get("think_repetition_penalty")
        think_max_new_tokens = gen_cfg.get("think_max_new_tokens")
        answer_max_new_tokens = gen_cfg.get("answer_max_new_tokens")
        answer_stop = gen_cfg.get("answer_stop")
        TIMEOUT_SEC = gen_cfg.get("timeout", 280)  # overall internal timeout per prompt (soft used below)

        tasks = []
        for p in prompts:
            coro = self._two_phase_for_one_prompt(
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
            async def run_with_timeout(coro=coro, prompt=p):
                try:
                    return await asyncio.wait_for(coro, timeout=TIMEOUT_SEC)
                except asyncio.TimeoutError:
                    return []
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return []
            tasks.append(asyncio.create_task(run_with_timeout()))

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise

        normalized: List[List[tuple[str, int, float]]] = []
        empty_all = True
        for r in results:
            if isinstance(r, Exception):
                normalized.append([])
            else:
                normalized.append(r)
                if r:
                    empty_all = False
        if empty_all:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0
        if self._consecutive_failures >= self.circuit_breaker_failures:
            self.metrics["circuit_breaker_trips"] += 1
            self._consecutive_failures = 0
            # allow disabling circuit breaker raising via config
            if not gen_cfg.get("disable_circuit_breaker", False):
                raise EngineCircuitBreaker("Too many consecutive empty generations; circuit breaker tripped")
        return normalized


def build_sglang_engine(model_name: str, sglang_config: Optional[Dict[str, Any]] = None) -> AsyncSGLangEngineWrapper:
    return AsyncSGLangEngineWrapper(model_name=model_name, sglang_config=sglang_config)