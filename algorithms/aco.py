import numpy as np
import time
from core.distance import tour_length

class AntColonyOptimization:
    """
    Mejoras vs original:
      - Pre-calculo de eta^beta fuera del loop de construccion
      - Deposito de feromonas con np.add.at (sin loop ciudad por ciudad)
      - Limites tau_min / tau_max para estabilidad numerica
    """
    def __init__(self, D, n, n_ants=20, alpha=1.0, beta=5.0, rho=0.3, Q=100.0, seed=0):
        self.D = D; self.n = n; self.n_ants = n_ants
        self.alpha = alpha; self.beta = beta; self.rho = rho; self.Q = Q
        self.rng = np.random.default_rng(seed)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.eta = np.where(D > 0, 1.0 / D, 0.0)
        tau0 = 1.0 / (n * float(np.mean(D[D > 0])))
        self.tau = np.full((n, n), tau0, dtype=np.float64)
        self.tau_min, self.tau_max = 1e-10, 1e6

    def _construct(self, eta_b: np.ndarray) -> list:
        start = int(self.rng.integers(0, self.n))
        visited = np.zeros(self.n, dtype=bool)
        tour = [start]; visited[start] = True
        for _ in range(self.n - 1):
            cur = tour[-1]
            desire = (self.tau[cur] ** self.alpha) * eta_b[cur]
            desire[visited] = 0.0
            total = desire.sum()
            if total == 0.0:
                nxt = int(self.rng.choice(np.where(~visited)[0]))
            else:
                nxt = int(self.rng.choice(self.n, p=desire/total))
            tour.append(nxt); visited[nxt] = True
        return tour

    def _update(self, tours: list, lengths: list):
        self.tau *= (1.0 - self.rho)
        for tour, length in zip(tours, lengths):
            t = np.asarray(tour)
            delta = self.Q / length
            i_arr, j_arr = t, np.roll(t, -1)
            np.add.at(self.tau, (i_arr, j_arr), delta)
            np.add.at(self.tau, (j_arr, i_arr), delta)
        np.clip(self.tau, self.tau_min, self.tau_max, out=self.tau)

    def run(self, budget_seconds: float):
        best_len = float('inf')
        history = []
        t0 = time.time()
        while time.time() - t0 < budget_seconds:
            eta_b = self.eta ** self.beta          # pre-calculo por iteracion
            tours = [self._construct(eta_b) for _ in range(self.n_ants)]
            lengths = [tour_length(t, self.D) for t in tours]
            it_best = min(lengths)
            if it_best < best_len: best_len = it_best
            self._update(tours, lengths)
            history.append(best_len)
        return best_len, history


def aco_solver(D, n, seed, budget):
    return AntColonyOptimization(D, n, n_ants=20, alpha=1.0, beta=5.0,
                                 rho=0.3, Q=100.0, seed=seed).run(budget)