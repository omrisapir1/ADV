from __future__ import annotations

import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple
import re
import time
import math
import json

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
        base_url = "http://localhost:30000/v1"
        api_key  = "EMPTY"
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.base_url = base_url  # for health checks / manual fallbacks

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
            "fallback_http_used": 0,
            "empty_choices": 0,
        }

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self.metrics)

    def hot_swap(self, tmp_weights_path: str):
        url = f"{self.base_url.rsplit('/',1)[0]}/update_weights_from_disk"
        data = {"model_path": tmp_weights_path}
        try:
            requests.post(url, json=data, timeout=10)
        except Exception:
            pass

    def health_check(self) -> bool:
        """Check server health by hitting /models; fallback to /completions echo."""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        try:
            payload = {
                "model": self.model_name,
                "prompt": "ping",
                "max_tokens": 1,
                "temperature": 0,
            }
            resp = requests.post(f"{self.base_url}/completions", json=payload, timeout=3)
            return resp.status_code == 200 and 'choices' in resp.json()
        except Exception:
            return False

    async def _http_fallback_completion(self, payload: Dict[str, Any]) -> Any:
        """Fallback: direct HTTP POST to /completions only (removed chat fallback)."""
        def _do_post(url: str, data: Dict[str, Any]):
            try:
                r = requests.post(url, json=data, timeout=self.per_request_timeout)
                return r.status_code, r.text
            except Exception as e:
                return 599, str(e)
        comp_url = f"{self.base_url}/completions"
        status_c, text_c = await asyncio.to_thread(_do_post, comp_url, payload)
        parsed_c = None
        try:
            parsed_c = json.loads(text_c)
        except Exception:
            pass
        if status_c == 200 and isinstance(parsed_c, dict) and parsed_c.get('choices'):
            self.metrics['fallback_http_used'] += 1
            return parsed_c
        print("[generation][FALLBACK] HTTP /completions fallback failed.")
        print(f"[generation][FALLBACK] status={status_c} body_head={text_c[:200]}")
        class Dummy: choices = []
        return Dummy()

    async def _completion_call(self, **kwargs):
        async with self._semaphore:
            self.metrics["in_flight"] += 1
            start = time.monotonic()
            task = asyncio.create_task(self.client.completions.create(**kwargs))
            try:
                resp = await asyncio.wait_for(task, timeout=self.per_request_timeout)
                self.metrics["total_requests"] += 1
                if not getattr(resp, 'choices', None):
                    self.metrics['empty_choices'] += 1
                    print("[generation][WARN] Empty choices from SDK; attempting HTTP fallback.")
                    prompt_txt = (kwargs.get("prompt") or "")[:160]
                    print(f"[generation][WARN] Prompt_head: {prompt_txt!r}")
                    fallback_payload = {
                        "model": kwargs.get("model", self.model_name),
                        "prompt": kwargs.get("prompt"),
                        "max_tokens": kwargs.get("max_tokens"),
                        "temperature": kwargs.get("temperature"),
                        "top_p": kwargs.get("top_p"),
                        "n": kwargs.get("n", 1),
                        "stop": kwargs.get("stop"),
                    }
                    fb = await self._http_fallback_completion(fallback_payload)
                    if isinstance(fb, dict) and fb.get('choices'):
                        class _Resp:
                            def __init__(self, raw):
                                self.choices = raw['choices']
                        print("[generation][INFO] HTTP fallback succeeded; using fallback response.")
                        return _Resp(fb)
                return resp
            except asyncio.TimeoutError:
                self.metrics["timeouts"] += 1
                print("[generation] Timeout in _completion_call")
                class Dummy: choices = []
                return Dummy()
            except Exception as e:
                import traceback
                self.metrics["errors"] += 1
                print("[generation] ERROR in _completion_call:", e)
                traceback.print_exc()
                class Dummy: choices = []
                return Dummy()
            finally:
                self.metrics["total_time"] += (time.monotonic() - start)
                self.metrics["in_flight"] -= 1

    def _entropy_and_explore_from_top_logprobs(self, top_lp: Dict[Any, float], token_lp: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        probs = [math.exp(lp) for lp in top_lp.values()]
        Z = sum(probs)
        if Z <= 0 or not math.isfinite(Z):
            return None, None
        probs = [p / Z for p in probs]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        try:
            p_selected = math.exp(token_lp) / Z if token_lp is not None else None
        except Exception:
            p_selected = None
        return entropy, p_selected

    def _extract_choice_text(self, choice: Any) -> str:
        if choice is None:
            return ""
        txt = getattr(choice, "text", None)
        if isinstance(txt, str) and txt:
            return txt
        if isinstance(choice, dict) and isinstance(choice.get("text"), str):
            return choice["text"]
        return ""

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
    ) -> List[tuple[str, int, float, float]]:
        payload_extra_1 = {"top_k": think_top_k, "repetition_penalty": think_repetition_penalty}
        try:
            resp1 = await self._completion_call(
                model=self.model_name,
                prompt=base_prompt,
                n=n_samples,
                temperature=think_temperature,
                top_p=think_top_p,
                max_tokens=think_max_new_tokens,
                stop=[THINK_STOP],
                extra_body=payload_extra_1,
                logprobs=20,
            )
        except Exception as e:
            print(f"[generation] Phase1 fatal error for prompt: {e}")
            return []
        choices_phase1 = getattr(resp1, "choices", [])
        if not choices_phase1:
            print("[generation][WARN] Phase1 produced zero choices.")
        results: List[tuple[str, int, float, float]] = [("", 0, None, None)] * len(choices_phase1)
        phase2_items: List[Tuple[int, str, str, Optional[float], Optional[float]]] = []
        for idx, choice in enumerate(choices_phase1):
            think_piece = self._extract_choice_text(choice)
            finish_reason = getattr(choice, "finish_reason", None)
            avg_entropy: Optional[float] = None
            avg_p_selected: Optional[float] = None
            lp_obj = getattr(choice, "logprobs", None)
            top_lps = getattr(lp_obj, "top_logprobs", None) or []
            token_lps = getattr(lp_obj, "token_logprobs", None) or []
            entropies: List[float] = []
            ps_selected: List[float] = []
            for top_lp, token_lp in zip(top_lps, token_lps):
                h, p_sel = self._entropy_and_explore_from_top_logprobs(top_lp, token_lp)
                if h is not None:
                    entropies.append(h)
                if p_sel is not None:
                    ps_selected.append(p_sel)
            if entropies:
                avg_entropy = sum(entropies) / len(entropies)
            if ps_selected:
                avg_p_selected = sum(ps_selected) / len(ps_selected)
            if finish_reason != "stop" or re.findall(r"\\boxed\s*{(.*?)}", think_piece or "", flags=re.DOTALL):
                results[idx] = (think_piece, 0, avg_entropy, avg_p_selected)
                continue
            think_clean = think_piece.split(THINK_STOP, 1)[0] if THINK_STOP in think_piece else think_piece
            context = base_prompt + think_clean + THINK_STOP
            phase2_items.append((idx, think_clean, context, avg_entropy, avg_p_selected))
        if not phase2_items:
            return results
        payload_extra_2 = {"top_k": 0, "repetition_penalty": 1.0}
        async def _greedy(ctx: str):
            return await self._completion_call(
                model=self.model_name,
                prompt=ctx,
                n=1,
                temperature=0.0,
                top_p=1.0,
                max_tokens=answer_max_new_tokens,
                stop=answer_stop if answer_stop else None,
                extra_body=payload_extra_2,
            )
        import traceback
        try:
            for start_idx in range(0, len(phase2_items), self.phase2_batch_limit):
                batch = phase2_items[start_idx:start_idx + self.phase2_batch_limit]
                tasks = [asyncio.create_task(_greedy(ctx)) for _, _, ctx, _, _ in batch]
                self.metrics["phase2_batches"] += 1
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                for (idx, think_clean, _ctx, avg_entropy, avg_p_selected), resp2 in zip(batch, gathered):
                    if isinstance(resp2, Exception) or not getattr(resp2, "choices", None):
                        full_text = think_clean + THINK_STOP
                        results[idx] = (full_text, 0, avg_entropy, avg_p_selected)
                        continue
                    ans_choice = resp2.choices[0] if resp2.choices else None
                    answer_text = self._extract_choice_text(ans_choice)
                    full_text = think_clean + THINK_STOP + answer_text
                    results[idx] = (full_text, 1, avg_entropy, avg_p_selected)
                await asyncio.sleep(random.uniform(0.005, 0.02))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print("[generation] Phase2 error:", e)
            traceback.print_exc()
        return results

    async def generate_candidates(
        self,
        prompts: List[str],
        n_samples: int,
        **gen_cfg: Any,
    ) -> List[List[tuple[str, int, float, float]]]:
        think_temperature = gen_cfg.get("think_temperature")
        think_top_p = gen_cfg.get("think_top_p")
        think_top_k = gen_cfg.get("think_top_k")
        think_repetition_penalty = gen_cfg.get("think_repetition_penalty")
        think_max_new_tokens = gen_cfg.get("think_max_new_tokens")
        answer_max_new_tokens = gen_cfg.get("answer_max_new_tokens")
        answer_stop = gen_cfg.get("answer_stop")
        TIMEOUT_SEC = gen_cfg.get("timeout", 500)
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
                    print(f"[generation] Prompt timeout: {prompt[:60]}...")
                    return []
                except Exception as e:
                    import traceback
                    print(f"[generation] Prompt error for '{prompt[:60]}...':", e)
                    traceback.print_exc()
                    return []
            tasks.append(asyncio.create_task(run_with_timeout()))
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            import traceback
            traceback.print_exc()
            raise
        normalized: List[List[tuple[str, int, float, float]]] = []
        empty_all = True
        for r in results:
            if isinstance(r, Exception):
                print("[generation] Gathered exception (suppressed):", r)
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
            raise EngineCircuitBreaker("Too many consecutive empty generations; circuit breaker tripped")
        return normalized


def build_sglang_engine(model_name: str, sglang_config: Optional[Dict[str, Any]] = None) -> AsyncSGLangEngineWrapper:
    return AsyncSGLangEngineWrapper(model_name=model_name, sglang_config=sglang_config)