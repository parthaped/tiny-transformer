import torch
from typing import List

class Replicator:
    """
    Discrete-time replicator dynamics with optional mutation (replicator–mutator).
      x_{t+1,i} = (1 - mu) * x_{t,i} * u_i / (x_t · u) + mu * 1/H
    """
    def __init__(self, num_heads: int, mutation: float = 0.0, eps: float = 1e-8):
        self.H = num_heads
        self.mu = float(mutation)
        self.eps = eps
        self.history: List[torch.Tensor] = []

    def step(self, x: torch.Tensor, payoffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: current population distribution (H,), nonnegative, sums to 1
            payoffs: payoff vector (H,)
        Returns:
            x_next: updated distribution (H,)
        """
        assert x.ndim == 1 and payoffs.ndim == 1
        assert x.shape[0] == self.H == payoffs.shape[0]

        x = torch.clamp(x, min=0.0)
        if x.sum() <= self.eps:
            x = torch.ones_like(x) / self.H

        mean_payoff = torch.dot(x, payoffs).clamp(min=self.eps)
        x_repl = x * (payoffs / mean_payoff)
        x_repl = x_repl / x_repl.sum().clamp(min=self.eps)

        if self.mu > 0.0:
            uniform = torch.ones_like(x) / self.H
            x_next = (1 - self.mu) * x_repl + self.mu * uniform
        else:
            x_next = x_repl

        x_next = torch.clamp(x_next, min=0.0)
        x_next = x_next / x_next.sum().clamp(min=self.eps)

        self.history.append(x_next.detach().cpu())
        return x_next

    def get_history(self) -> torch.Tensor:
        if not self.history:
            return torch.empty(0)
        return torch.stack(self.history, dim=0)  # (T, H)
