import numpy as np
import time
from core.distance import tour_length


class AntColonyOptimization:
    """
    Ant-System clasico para TSP.

    Optimizaciones aplicadas:
      - eta_b = eta^beta precalculado en __init__ (constante, no cambia nunca)
      - alpha==1.0: evita tau**alpha innecesario en _construct
      - Deposito de feromonas con np.add.at (sin loop ciudad por ciudad)
      - Evaporacion vectorial con *= en lugar de loop
      - np.clip in-place para estabilidad numerica (tau_min / tau_max)

    Bugs corregidos vs version anterior:
      - eta_b se recalculaba en cada iteracion del while siendo siempre identico
      - tau**alpha se calculaba incluso cuando alpha=1.0 (potencia innecesaria ~2x)
    """

    def __init__(self, D, n, n_ants=20, alpha=1.0, beta=5.0,
                 rho=0.3, Q=100.0, seed=0):
        self.D       = D
        self.n       = n
        self.n_ants  = n_ants
        self.alpha   = alpha
        self.beta    = beta
        self.rho     = rho
        self.Q       = Q
        self.rng     = np.random.default_rng(seed)

        # Heuristica eta = 1/d (0 en diagonal)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.eta = np.where(D > 0, 1.0 / D, 0.0)

        # FIX: precalcular eta_b UNA sola vez — es constante durante toda la ejecucion
        self.eta_b = self.eta ** self.beta

        # Feromonas iniciales uniformes
        tau0 = 1.0 / (n * float(np.mean(D[D > 0])))
        self.tau = np.full((n, n), tau0, dtype=np.float64)

        # Limites para estabilidad numerica
        self.tau_min = 1e-10
        self.tau_max = 1e6

    def _construct(self) -> list:
        """
        Construye una solucion para una hormiga.
        Usa self.eta_b precalculado en lugar de recibirlo como parametro.
        FIX: evita tau**alpha cuando alpha==1.0 (ahorra ~2x en esa operacion).
        """
        start   = int(self.rng.integers(0, self.n))
        visited = np.zeros(self.n, dtype=bool)
        tour    = [start]
        visited[start] = True

        for _ in range(self.n - 1):
            cur = tour[-1]

            # FIX: evitar potencia innecesaria cuando alpha == 1.0
            tau_row = self.tau[cur] if self.alpha == 1.0 else self.tau[cur] ** self.alpha
            desire  = tau_row * self.eta_b[cur]   # nuevo array — no corrompe tau
            desire[visited] = 0.0

            total = desire.sum()
            if total == 0.0:
                nxt = int(self.rng.choice(np.where(~visited)[0]))
            else:
                nxt = int(self.rng.choice(self.n, p=desire / total))

            tour.append(nxt)
            visited[nxt] = True

        return tour

    def _update(self, tours: list, lengths: list):
        """
        Actualiza feromonas: evaporacion + deposito simetrico.
        np.add.at garantiza deposito correcto con indices repetidos.
        """
        # Evaporacion vectorial
        self.tau *= (1.0 - self.rho)

        # Deposito proporcional a calidad del tour
        for tour, length in zip(tours, lengths):
            t     = np.asarray(tour)
            delta = self.Q / length
            i_arr = t
            j_arr = np.roll(t, -1)
            np.add.at(self.tau, (i_arr, j_arr), delta)
            np.add.at(self.tau, (j_arr, i_arr), delta)

        # Clip para estabilidad numerica
        np.clip(self.tau, self.tau_min, self.tau_max, out=self.tau)

    def run(self, budget_seconds: float):
        best_len = float('inf')
        history  = []
        t0       = time.time()

        while time.time() - t0 < budget_seconds:
            # FIX: ya no se recalcula eta_b aqui — se usa self.eta_b del __init__
            tours   = [self._construct() for _ in range(self.n_ants)]
            lengths = [tour_length(t, self.D) for t in tours]

            it_best = min(lengths)
            if it_best < best_len:
                best_len = it_best

            self._update(tours, lengths)
            history.append(best_len)

        return best_len, history


def aco_solver(D, n, seed, budget):
    return AntColonyOptimization(
        D, n, n_ants=20, alpha=1.0, beta=5.0,
        rho=0.3, Q=100.0, seed=seed
    ).run(budget)