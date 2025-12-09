from typing import Dict, Any, List, Tuple
import json
import os

class AlphaControl:
    """Encapsulates alpha scheduling parameters and simple tracking state.

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
        # Store raw cfg via direct indexing per request
        self.initial_alpha: float = float(cfg["initial_alpha"])  # no .get
        self.alpha_step: float = float(cfg["alpha_step"])        # no .get
        self.avg_last_steps: int = int(cfg["avg_last_steps"])    # no .get
        self.adjust_every: int = int(cfg["adjust_every"])        # no .get
        self.correctness_improve_eps: float = float(cfg["correctness_improve_eps"])  # no .get
        self.pass1_improve_eps: float = float(cfg["pass1_improve_eps"])              # no .get
        self.entropy_change_eps: float = float(cfg["entropy_change_eps"])            # no .get
        self.alpha_explore_state: float = float(cfg["alpha_explore_state"])  # no .get
        self.max_alpha_step: float = float(cfg["max_alpha_step"])  # no .get


        # Runtime state (can be used later)
        self.alpha: float = self.initial_alpha
        self.correctness_history: List[float] = []
        self.pass1_history: List[float] = []
        self.entropy_history: List[float] = []

        self.last_correctness_avg: float = 0.0
        self.last_pass1_avg: float = 0.0
        self.last_entropy_avg: float = 0.0

    def step(self, new_correctness: float, new_pass1: float, new_entropy: float, step: int) -> None:
        """Update histories with new metrics."""
        self.correctness_history.append(new_correctness)
        self.pass1_history.append(new_pass1)
        self.entropy_history.append(new_entropy)
        if step % self.adjust_every != 0 or step==0:
            return
        # Compute averages over last avg_last_steps
        def avg_last(history: List[float]) -> float:
            relevant = history[-self.avg_last_steps:]
            return sum(relevant) / len(relevant) if relevant else 0.0
        correctness_avg = avg_last(self.correctness_history)
        pass1_avg = avg_last(self.pass1_history)
        entropy_avg = avg_last(self.entropy_history)

        if step == self.adjust_every:
            # First adjustment, just store averages
            self.last_correctness_avg = correctness_avg
            self.last_pass1_avg = pass1_avg
            self.last_entropy_avg = entropy_avg
            return
        print(f'last_correctness_avg: {self.last_correctness_avg}, correctness_avg: {correctness_avg}')
        print(f'last_pass1_avg: {self.last_pass1_avg}, pass1_avg: {pass1_avg}')
        print(f'last_entropy_avg: {self.last_entropy_avg}, entropy_avg: {entropy_avg}')
        if ((correctness_avg - self.last_correctness_avg) > self.correctness_improve_eps and
                (pass1_avg - self.last_pass1_avg) > self.pass1_improve_eps):
            print(f'Alpha remains the same - {self.alpha}')
        elif (entropy_avg - self.last_entropy_avg) > self.entropy_change_eps:
            print('Alpha decreased due to entropy improvement.')
            alpha_step = self.alpha_step
            self.alpha = max(0.0, self.alpha - alpha_step)
            print(f'Adjusted alpha to {self.alpha}')
        else:
            print('Alpha increased due to lack of improvement.')
            self.alpha = min(1.0, self.alpha + self.alpha_step)
            print(f'Adjusted alpha to {self.alpha}')
        # Update last averages
        self.last_correctness_avg = correctness_avg
        self.last_pass1_avg = pass1_avg
        self.last_entropy_avg = entropy_avg

    # -------------------- Persistence helpers --------------------
    def to_state(self) -> Dict[str, Any]:
        """Serialize runtime state for checkpointing."""
        return {
            "alpha": self.alpha,
            "correctness_history": self.correctness_history,
            "pass1_history": self.pass1_history,
            "entropy_history": self.entropy_history,
            "last_correctness_avg": self.last_correctness_avg,
            "last_pass1_avg": self.last_pass1_avg,
            "last_entropy_avg": self.last_entropy_avg,
        }

    def save_state(self, path: str) -> None:
        """Save runtime state to a JSON file at path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_state(), f)

    def load_state(self, path: str) -> None:
        """Load runtime state from JSON file; silently ignore if missing."""
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
        self.last_entropy_avg = float(data.get("last_entropy_avg", 0.0))
