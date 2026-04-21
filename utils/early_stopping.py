"""Early stopping utility for all solvers.

Stops training when loss plateaus. Prevents wasted iterations
when the model has converged.

Usage:
    es = EarlyStopping(patience=500, min_delta=1e-6, warmup=1000)
    for step in range(n_iter):
        loss = train_step(...)
        if es(loss.item()):
            print(f"Early stopping at step {step} (best={es.best_loss:.4e})")
            break
"""


class EarlyStopping:
    """Detects training plateau and signals when to stop.

    Parameters
    ----------
    patience : int
        Steps without improvement before stopping. Default 500.
    min_delta : float
        Minimum decrease in loss to count as improvement.
    warmup : int
        Don't stop before this many steps (gives training time to settle).
    mode : str
        "min" (loss) or "max" (for metrics to maximise).

    Attributes
    ----------
    best_loss : float
        Best (lowest) loss seen so far.
    counter : int
        Steps since last improvement.
    stopped : bool
        True once early stopping has triggered.
    """

    def __init__(self, patience=500, min_delta=1e-6, warmup=1000, mode="min"):
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got {mode}"
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.mode = mode
        self.best_loss = float('inf') if mode == "min" else -float('inf')
        self.counter = 0
        self.step = 0
        self.stopped = False

    def _is_improvement(self, loss):
        if self.mode == "min":
            return loss < self.best_loss - self.min_delta
        else:
            return loss > self.best_loss + self.min_delta

    def __call__(self, loss):
        """Update state. Returns True if training should stop."""
        self.step += 1

        if self._is_improvement(loss):
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        # Don't stop during warmup
        if self.step < self.warmup:
            return False

        if self.counter >= self.patience:
            self.stopped = True
            return True
        return False

    def reset(self):
        self.best_loss = float('inf') if self.mode == "min" else -float('inf')
        self.counter = 0
        self.step = 0
        self.stopped = False
