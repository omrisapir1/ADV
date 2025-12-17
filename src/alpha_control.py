from typing import Dict, Any, List
import json
import os


class EMA:
    def __init__(self, decay: float):
        self.decay = decay
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * x
        return self.value


class AlphaControl:
    """
    Alpha controller for explore–exploit tradeoff.

    Design:
    - Entropy is treated as a *stability signal* → tracked via EMA
    - Correctness & pass@1 are *progress signals* → tracked via rolling avg
    - Alpha is adjusted every `adjust_every` steps

    Config contract (alpha_control section):
    - initial_alpha: float
    - alpha_step: float
    - avg_last_steps: int
    - adjust_every: int
    - correctness_improve_eps: float
    - pass1_improve_eps: float
    - entropy_change_eps: float
    """

    def __init__(self, cfg: Dict[str, Any]):
        # ----- Config -----
        self.initial_alpha: float = float(cfg["initial_alpha"])
        self.alpha_step: float = float(cfg["alpha_step"])
        self.avg_last_steps: int = int(cfg["avg_last_steps"])
        self.adjust_every: int = int(cfg["adjust_every"])
        self.correctness_improve_eps: float = float(cfg["correctness_improve_eps"])
        self.pass1_improve_eps: float = float(cfg["pass1_improve_eps"])
        self.entropy_change_eps: float = float(cfg["entropy_change_eps"])
        self.update_ref_for_stuck_steps: int = int(cfg["update_ref_for_stuck_steps"])
        self.max_not_improved_steps: int = int(cfg["max_not_improved_steps"])

        # ----- Runtime state -----
        self.alpha: float = self.initial_alpha

        self.correctness_history: List[float] = []
        self.pass1_history: List[float] = []
        self.entropy_history: List[float] = []

        self.last_correctness_avg: float = 0.0
        self.last_pass1_avg: float = 0.0
        self.last_entropy_ema: float = 0.0

        # EMA for entropy only
        decay = 0.5# 1.0 - 1.0 / max(1, self.avg_last_steps)
        self.entropy_ema = EMA(decay)
        self.stuck_on_explore = 0
        self.not_improved_steps = 0

    # ------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------
    def step(
        self,
        new_correctness: float,
        new_pass1: float,
        new_entropy: float,
        step: int,
    ) -> bool:
        """
        Update alpha based on:
        - entropy EMA (instability)
        - rolling avg correctness
        - rolling avg pass@1
        """

        # ---- Track raw history ----
        self.correctness_history.append(new_correctness)
        self.pass1_history.append(new_pass1)
        self.entropy_history.append(new_entropy)
        self.entropy_ema.update(new_entropy)

        if step == 0 or step % self.adjust_every != 0:
            return False

        # ---- Rolling averages for progress signals ----
        def avg_last(history: List[float]) -> float:
            h = history[-self.avg_last_steps:]
            return sum(h) / len(h) if h else 0.0

        correctness_avg = avg_last(self.correctness_history)
        pass1_avg = avg_last(self.pass1_history)

        # ---- EMA for entropy (stability signal) ----
        entropy_ema = self.entropy_ema.update(new_entropy)

        # First adjustment: just initialize baselines
        if step == self.adjust_every:
            self.last_correctness_avg = correctness_avg
            self.last_pass1_avg = pass1_avg
            self.last_entropy_ema = entropy_ema
            return False

        # ---- Deltas ----
        correctness_delta = correctness_avg - self.last_correctness_avg
        pass1_delta = pass1_avg - self.last_pass1_avg
        entropy_delta = entropy_ema - self.last_entropy_ema

        print(
            f"[AlphaControl] step={step} | α={self.alpha:.3f}\n"
            f"  correctness_avg={correctness_avg:.4f} (Δ={correctness_delta:+.4f})\n"
            f"  pass1_avg={pass1_avg:.4f} (Δ={pass1_delta:+.4f})\n"
            f"  entropy_ema={entropy_ema:.4f} (Δ={entropy_delta:+.4f})"
        )

        # ------------------------------------------------------------
        # Control logic
        # ------------------------------------------------------------

        # 1. Entropy rising too fast → exploration becoming unstable
        if entropy_delta > self.entropy_change_eps:
            self.alpha = max(0.0, self.alpha - self.alpha_step)
            print("[AlphaControl] ↓ alpha (entropy rising)")
            self.stuck_on_explore = 0

        # 2. Learning progress detected → keep alpha
        elif (
            correctness_delta > self.correctness_improve_eps
            or pass1_delta > self.pass1_improve_eps
        ):
            self.stuck_on_explore = 0
            print("[AlphaControl] α unchanged (learning progress)")

        # 3. No progress & entropy stable → need more exploration
        elif self.alpha == 1.0:

            self.stuck_on_explore += 1

            print(f"[AlphaControl] alpha stuck at 1.0 stuck_on_explore={self.stuck_on_explore}")
        elif entropy_delta < (-self.entropy_change_eps) or self.not_improved_steps == self.max_not_improved_steps:
            self.alpha = min(1.0, self.alpha + self.alpha_step)
            print("[AlphaControl] ↑ alpha (no progress, stable entropy)")
            self.stuck_on_explore = 0
            self.not_improved_steps = 0
        else:
            self.not_improved_steps += 1


        # ---- Update baselines ----
        self.last_correctness_avg = correctness_avg
        self.last_pass1_avg = pass1_avg
        self.last_entropy_ema = entropy_ema

        print(f"[AlphaControl] new α = {self.alpha:.3f}")
        if self.stuck_on_explore == self.update_ref_for_stuck_steps:
            self.stuck_on_explore = 0
            print("[AlphaControl] resetting reference model")
            return True

        return False

    # ------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------
    def to_state(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "correctness_history": self.correctness_history,
            "pass1_history": self.pass1_history,
            "entropy_history": self.entropy_history,
            "last_correctness_avg": self.last_correctness_avg,
            "last_pass1_avg": self.last_pass1_avg,
            "last_entropy_ema": self.last_entropy_ema,
            "entropy_ema_value": self.entropy_ema.value,
        }

    def save_state(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_state(), f)

    def load_state(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        self.alpha = float(data.get("alpha", self.initial_alpha))
        self.correctness_history = list(data.get("correctness_history", []))
        self.pass1_history = list(data.get("pass1_history", []))
        self.entropy_history = list(data.get("entropy_history", []))

        self.last_correctness_avg = float(data.get("last_correctness_avg", 0.0))
        self.last_pass1_avg = float(data.get("last_pass1_avg", 0.0))
        self.last_entropy_ema = float(data.get("last_entropy_ema", 0.0))

        self.entropy_ema.value = data.get("entropy_ema_value", None)
