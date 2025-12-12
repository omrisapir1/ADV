import regex as re
from fractions import Fraction
from math import isclose
from typing import Union, Optional, List

from .constants import THINK_STOP

_COMMA = re.compile(",")
_PCT_END = re.compile(r"(?:\\?%)\s*$")
_UNICODE_MINUS = "\u2212"
THINK_TO_FAR_CHARS = 500

def _parse_digits(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip().replace(_UNICODE_MINUS, "-")
    s = _COMMA.sub("", s)
    # Mixed number: "a b/c"
    if re.fullmatch(r"[-+]?\d+\s+\d+\s*/\s*\d+", s):
        try:
            a, b = s.split(None, 1)
            return float(int(a) + Fraction(b.replace(" ", "")))
        except Exception:
            pass
    # Fraction: "a/b"
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", s):
        try:
            return float(Fraction(s.replace(" ", "")))
        except Exception:
            pass
    pct = False
    if _PCT_END.search(s):
        s = _PCT_END.sub("", s).strip()
        pct = True
    try:
        v = float(s)
        return v / 100.0 if pct else v
    except Exception:
        return None

def _numeric_equal(a: float, b: float, rel: float = 1e-4, abs_tol: float = 1e-12) -> bool:
    return isclose(a, b, rel_tol=rel, abs_tol=abs_tol)

def extract_final_answer(llm_output: str) -> tuple[str, bool]:
    """Fast extractor assuming only boxed answers or numeric last line."""
    pattern = re.compile(r"\\boxed\s*\{(.*?)\}", flags=re.DOTALL)
    matches = list(pattern.finditer(llm_output))
    last_think_end = llm_output.rfind(THINK_STOP)
    if matches:
        last_match = matches[-1]
        start_idx = last_match.start()
        can_only_be_zero = ('boxed' in llm_output[start_idx+5:].lower() or
                 last_think_end > start_idx or
                 last_think_end == -1 or
                 (start_idx - last_think_end) > THINK_TO_FAR_CHARS )
        return last_match.group(1).strip(),  can_only_be_zero
    lines = [ln.strip() for ln in (llm_output or "").splitlines() if ln.strip()]
    return lines[-1] if lines else None, last_think_end == -1

def math_equal(pred: Union[str, float, int, None],
               ref: Union[str, float, int, None],
               can_only_be_zero: bool = False,
               rel_tol: float = 1e-4) -> int:
    if pred is None or ref is None:
        return 0
    p_num = _parse_digits(str(pred))
    r_num = _parse_digits(str(ref))
    if p_num is None or r_num is None:
        if ''.join([c for c in pred if c.isdigit()]) == ref.replace('.','').replace('-',''):
            return -1
        return 0
    # Allow equality up to 1e-4 relative tolerance or exact match
    if _numeric_equal(p_num, r_num, rel=rel_tol) or \
           _numeric_equal(p_num, r_num * 100.0, rel=rel_tol) or \
           _numeric_equal(p_num, r_num / 100.0, rel=rel_tol):
        if can_only_be_zero:
            return -1
        return 1
    if ''.join([c for c in pred if c.isdigit()]) == ref.replace('.','').replace('-',''):
        return -1
    return 0


def compute_final_correctness(candidates: List[List[str]], gold_answers: List[str]) -> List[List[int]]:
    B = len(candidates)
    if len(gold_answers) != B:
        raise ValueError("gold_answers length must match candidates batch size")
    if B == 0:
        return []

    out: List[List[int]] = []
    for row, g in zip(candidates, gold_answers):
        row_flags: List[int] = []
        for cand_raw in row:
            cand_ans, can_only_be_zero = extract_final_answer(cand_raw)

            row_flags.append(math_equal(cand_ans, g, can_only_be_zero))
        out.append(row_flags)
    return out
