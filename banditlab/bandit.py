"""Beta-Bernoulli Thompson Sampling with two independent contexts."""

import numpy as np

CONTEXTS = ("cellular", "telephone")
ACTIONS = ("short_message", "guided_explanation", "benefits_summary")
MESSAGES = {
    "short_message": "Conheça a oferta de depósito a prazo e consulte as condições.",
    "guided_explanation": "Veja como funciona o depósito a prazo, seus prazos e condições.",
    "benefits_summary": "Confira os benefícios, riscos e condições do depósito a prazo.",
}


class ThompsonSampling:
    def __init__(self, seed: int = 42, alpha=None, beta=None):
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones((2, 3)) if alpha is None else np.array(alpha, dtype=float)
        self.beta = np.ones((2, 3)) if beta is None else np.array(beta, dtype=float)
        if self.alpha.shape != (2, 3) or self.beta.shape != (2, 3):
            raise ValueError("Posterior arrays must have shape (2, 3)")
        if not (np.isfinite(self.alpha).all() and np.isfinite(self.beta).all()):
            raise ValueError("Posterior parameters must be finite")
        if (self.alpha <= 0).any() or (self.beta <= 0).any():
            raise ValueError("Posterior parameters must be positive")

    @property
    def means(self):
        return self.alpha / (self.alpha + self.beta)

    def select(self, context: int) -> int:
        samples = self.rng.beta(self.alpha[context], self.beta[context])
        return int(np.argmax(samples))

    def update(self, context: int, action: int, reward: int):
        if reward not in (0, 1):
            raise ValueError("Reward must be binary")
        self.alpha[context, action] += reward
        self.beta[context, action] += 1 - reward

    def state(self) -> dict:
        return {"alpha": self.alpha.tolist(), "beta": self.beta.tolist()}


class EpsilonGreedy(ThompsonSampling):
    def __init__(self, seed: int = 42, epsilon: float = 0.1, **kwargs):
        super().__init__(seed, **kwargs)
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between zero and one")
        self.epsilon = epsilon

    def select(self, context: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(3))
        scores = self.means[context]
        return int(self.rng.choice(np.flatnonzero(scores == scores.max())))
