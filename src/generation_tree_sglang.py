from __future__ import annotations

import math
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Deque, Callable
from collections import deque

THINK_STOP = "</think>"

try:  # pragma: no cover
    from openai import AsyncOpenAI  # type: ignore
    from openai._exceptions import APIStatusError, RateLimitError  # type: ignore
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore
    class APIStatusError(Exception):  # type: ignore
        pass
    class RateLimitError(Exception):  # type: ignore
        pass

MAX_SPLIT = 4
MIN_SPLIT = 1

@dataclass
class _Node:
    ctx: str                # original prompt/context
    think: str              # accumulated think tokens (no THINK_STOP)
    depth: int              # depth in tree


class AsyncTreeOfThoughtSGLangEngineWrapper:
    """Merged Tree-of-Thought engine (sglang/openai stream) with vLLM-style branching logic.

    Preserves generate_candidates API from generation_tree.py.

    Branching differences vs original generation_tree:
      * Branching does NOT terminate the parent stream; parent continues generating.
      * When branching, we use candidate token logic similar to generation_vllm:
          cand_items = [(t, lp) for t, lp in top_candidates.items() if t != last_token]
        spawning one child with the chosen (already generated) token and one alternative sampled.
      * Single branching parameter set (entropy_threshold, token_prob_threshold, min_tokens_split).
        Accepts either legacy individual thresholds or split_param_sets (dict or list; first element used).


    Output format unchanged: List[List[(text, phase_flag)]] where phase_flag 0 means think only, 1 means think+answer.
    """

    def __init__(self, model_name: str, sglang_config: Optional[Dict[str, Any]] = None):
        if AsyncOpenAI is None:
            raise ImportError("openai async client unavailable; install openai")
        sglang_config = sglang_config or {}
        base_url = sglang_config.get("base_url", "http://localhost:30000/v1")
        api_key = sglang_config.get("api_key", "EMPTY")
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self._semaphore = asyncio.Semaphore(int(sglang_config.get("max_concurrency", 4)))
        self.max_retries = int(sglang_config.get("max_retries", 3))
        self.retry_sleep = float(sglang_config.get("retry_sleep", 5.0))

    # ---------------- entropy helper -----------------
    def _compute_entropy(self, top_logprob_items: dict) -> Tuple[float, List[Tuple[str, float]]]:
        probs: List[Tuple[str, float]] = []
        for token, lp in top_logprob_items.items():
            try:
                p = math.exp(lp)
            except Exception:
                p = 0.0
            probs.append((token, p))
        if not probs:
            return 0.0, []
        total = sum(p for _, p in probs) or 1e-12
        norm = [(t, p / total) for t, p in probs]
        entropy = -sum(p * math.log(p + 1e-12) for _, p in norm)
        return entropy, norm

    # --------------- streaming single node ---------------
    async def _stream_node(
        self,
        node: _Node,
        *,
        param_set: Dict[str, Any],
        think_max_new_tokens: int,
        min_global_tokens_split: int,
        token_prob_floor: float,
        max_depth: int,
        enqueue_child: Callable[[List[_Node]], None],
    ) -> str:
        """Stream think tokens for a node, spawning children inline without stopping parent.

        Returns final accumulated think string (may or may not include THINK_STOP; we do not force stop).
        Branch creation criteria evaluated using single param_set.
        """
        prompt = node.ctx + node.think
        try:
            async with self._semaphore:
                stream = await self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    n=1,
                    stream=True,
                    logprobs=5,
                    temperature=0.7,
                    top_p=0.7,
                    stop=[THINK_STOP],
                    max_tokens=think_max_new_tokens,
                    extra_body={"top_k": 40},
                )
        except Exception as e:
            print(f"[MergedTree] stream start error: {e}")
            return node.think

        token_count = 0
        accumulated = node.think
        # Keep generating until stop / budget; do NOT abort when branching.
        async for event in stream:
            choice = event.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            logprobs_obj = getattr(choice, "logprobs", None)
            tokens_list: List[str] = []
            top_logprobs_list: List[dict] = []
            # Extract token strings + top logprobs sequence
            if logprobs_obj is not None:
                if hasattr(logprobs_obj, "tokens") and hasattr(logprobs_obj, "top_logprobs"):
                    try:
                        tokens_list = list(getattr(logprobs_obj, "tokens") or [])
                        top_logprobs_list = list(getattr(logprobs_obj, "top_logprobs") or [])
                    except Exception:
                        pass
                elif isinstance(logprobs_obj, dict):
                    if "tokens" in logprobs_obj and isinstance(logprobs_obj.get("tokens"), list):
                        tokens_list = logprobs_obj.get("tokens") or []
                        top_logprobs_list = logprobs_obj.get("top_logprobs") or []
            if len(top_logprobs_list) != len(tokens_list):
                # Best effort: mismatch - treat no branching for this event
                top_logprobs_list = [{} for _ in tokens_list]
            for tok_idx, token_str in enumerate(tokens_list):
                if token_count >= think_max_new_tokens:
                    return accumulated
                prefix_before_token = accumulated
                accumulated += token_str
                top_lp_dict = top_logprobs_list[tok_idx] if tok_idx < len(top_logprobs_list) else {}
                entropy, norm_probs = self._compute_entropy(top_lp_dict) if top_lp_dict else (0.0, [])
                token_count += 1
                ent_thresh = param_set.get("entropy_threshold", 9999.0)
                prob_thresh = param_set.get("token_prob_threshold", token_prob_floor)
                min_split_tokens = param_set.get("min_tokens_split", min_global_tokens_split)

                if (
                    node.depth < max_depth
                    and entropy > ent_thresh
                    and token_count >= min_split_tokens
                    and norm_probs
                ) or (node.depth ==0 and token_count==1):
                    node.depth += 1
                    token_count = 0
                    # Build children: ONLY alternative tokens (parent continues with last_token itself)
                    try:
                        last_token = token_str
                        # Normalize probability list already in norm_probs; exclude last_token
                        alt_candidates = [(t, p) for t, p in norm_probs if t != last_token]
                        # Sort by descending probability for deterministic selection order
                        alt_candidates.sort(key=lambda x: x[1], reverse=True)
                        children: List[_Node] = []

                        for i, (tok_candidate, prob) in enumerate(alt_candidates):
                            if (prob >= prob_thresh and i < MAX_SPLIT) or i < (MIN_SPLIT + int(node.depth ==0) ):

                                children.append(
                                    _Node(
                                        ctx=node.ctx,
                                        think=prefix_before_token + tok_candidate,
                                        depth=node.depth + 1,
                                    )
                                )

                        if children:
                            enqueue_child(children)
                    except Exception as be:
                        print(f"[MergedTree] branching error: {be}")
                # Continue parent regardless of branching
            if finish_reason == "stop" or token_count >= think_max_new_tokens:
                return accumulated
        return accumulated

    # --------------- answer phase (unchanged) ---------------
    async def _greedy_answer(self, base_prompt_plus_think: str, answer_max_new_tokens: int, answer_stop: Optional[List[str]]) -> str:
        prompt = base_prompt_plus_think + THINK_STOP
        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    resp = await self.client.completions.create(
                        model=self.model_name,
                        prompt=prompt,
                        n=1,
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=answer_max_new_tokens,
                        stop=answer_stop if answer_stop else None,
                    )
                choice = resp.choices[0] if resp.choices else None
                if choice is None:
                    return ""
                return getattr(choice, "text", "") or ""
            except (APIStatusError, RateLimitError) as e:
                print(f"[MergedTree] answer API error {e}; retry {attempt+1}/{self.max_retries}")
                await asyncio.sleep(self.retry_sleep)
            except Exception as e:
                print(f"[MergedTree] answer generic error {e}; retry {attempt+1}/{self.max_retries}")
                await asyncio.sleep(self.retry_sleep)
        return ""

    # --------------- public API (unchanged signature) ---------------
    async def generate_candidates(self, prompts: List[str], **gen_cfg: Any) -> List[List[tuple[str, int]]]:
        # Legacy single thresholds (only source now)
        legacy_entropy = gen_cfg.get("think_entropy_threshold")
        legacy_prob = gen_cfg.get("token_prob_threshold")
        legacy_min_split = gen_cfg.get("min_tokens_split") or gen_cfg.get("think_min_tokens_split")  # allow alias

        # Simplified: always one param_set derived from legacy params; removed split_param_sets support
        param_set = {
            "entropy_threshold": legacy_entropy if legacy_entropy is not None else 0.4,
            "token_prob_threshold": legacy_prob if legacy_prob is not None else 0.05,
            "min_tokens_split": legacy_min_split if legacy_min_split is not None else 15,
        }

        think_max_new_tokens = gen_cfg.get("think_max_new_tokens")
        answer_max_new_tokens = gen_cfg.get("answer_max_new_tokens")
        answer_stop = gen_cfg.get("answer_stop")
        max_depth = int(gen_cfg.get("max_depth", 7))
        if think_max_new_tokens is None or answer_max_new_tokens is None:
            raise ValueError("think_max_new_tokens and answer_max_new_tokens must be provided in gen_cfg")


        async def _process_prompt(p: str) -> List[tuple[str, int]]:
            results: List[tuple[str, int]] = []
            queue: Deque[_Node] = deque([_Node(ctx=p, think="", depth=0)])
            pending_answer_tasks: List[asyncio.Task] = []
            pending_answer_meta: List[Tuple[int, str]] = []
            active: List[Tuple[asyncio.Task, _Node]] = []

            def enqueue_children(children: List[_Node]):
                if not children:
                    return
                # Start streaming immediately for all children; rely on semaphore for API concurrency
                for c in children:
                    task = asyncio.create_task(
                        self._stream_node(
                            c,
                            param_set=param_set,
                            think_max_new_tokens=think_max_new_tokens,
                            min_global_tokens_split=0,
                            token_prob_floor=0.0,
                            max_depth=max_depth,
                            enqueue_child=enqueue_children,
                        )
                    )
                    active.append((task, c))

            while queue or active:
                # Drain only the nodes that were queued at loop start (children added during processing start next iteration)
                if queue:
                    initial_queue_len = len(queue)
                    for _ in range(initial_queue_len):
                        node = queue.popleft()
                        task = asyncio.create_task(
                            self._stream_node(
                                node,
                                param_set=param_set,
                                think_max_new_tokens=think_max_new_tokens,
                                min_global_tokens_split=0,
                                token_prob_floor=0.0,
                                max_depth=max_depth,
                                enqueue_child=enqueue_children,
                            )
                        )
                        active.append((task, node))
                if not active:
                    break
                done, _pending = await asyncio.wait([t for t,_ in active], return_when=asyncio.FIRST_COMPLETED)
                for finished in done:
                    idx = next((i for i,(t,_) in enumerate(active) if t is finished), None)
                    if idx is None:
                        continue
                    node_obj = active[idx][1]
                    del active[idx]
                    try:
                        finished_think = finished.result()
                    except Exception as e:
                        print(f"[MergedTree] node stream error: {e}")
                        finished_think = node_obj.think
                    if THINK_STOP in finished_think:
                        think_clean = finished_think.split(THINK_STOP,1)[0]
                        pending_answer_tasks.append(asyncio.create_task(
                            self._greedy_answer(node_obj.ctx + think_clean, answer_max_new_tokens, answer_stop)
                        ))
                        pending_answer_meta.append((len(results), think_clean))
                        results.append((think_clean + THINK_STOP, 1))
                    else:
                        results.append((finished_think, 0))
            if pending_answer_tasks:
                answers = await asyncio.gather(*pending_answer_tasks)
                for (res_idx, think_clean), ans in zip(pending_answer_meta, answers):
                    full_text = think_clean + THINK_STOP + (ans or "")
                    results[res_idx] = (full_text, 1)
            return results

        tasks = [asyncio.create_task(_process_prompt(p)) for p in prompts]
        return await asyncio.gather(*tasks)


def build_tree_sglang_engine(model_name: str, sglang_config: Optional[Dict[str, Any]] = None) -> AsyncTreeOfThoughtSGLangEngineWrapper:
    return AsyncTreeOfThoughtSGLangEngineWrapper(model_name=model_name, sglang_config=sglang_config)
