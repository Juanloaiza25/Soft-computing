import random
import numpy as np
import time
from core.distance import tour_length, nearest_neighbor, two_opt_fast

def edge_diversity_fast(tour1: list, tour2: list) -> float:
    """
    OPTIMIZACION: tuplas (min,max) en lugar de frozenset(frozenset(...)).
    Construccion del set ~2x mas rapida.
    """
    n = len(tour1)
    def edges(t):
        return {(min(t[i], t[(i+1)%n]), max(t[i], t[(i+1)%n])) for i in range(n)}
    e1, e2 = edges(tour1), edges(tour2)
    u = len(e1 | e2)
    return 0.0 if u == 0 else 1.0 - len(e1 & e2) / u


class ChuBeasleyGA:
    def __init__(self, D, n, pop_size=60, pc=0.85, pm=0.05,
                 diversity_threshold=0.2, use_2opt=True, seed=0):
        self.D = D; self.n = n; self.pop_size = pop_size
        self.pc = pc; self.pm = pm; self.div_thresh = diversity_threshold
        self.use_2opt = use_2opt; self.rng = random.Random(seed)

    def _hash(self, tour):
        m = min(range(self.n), key=lambda i: tour[i])
        return tuple(tour[m:] + tour[:m])

    def _init_pop(self):
        pop, fits, hashes = [], [], set()
        for start in range(min(self.pop_size, self.n)):
            t = nearest_neighbor(self.D, start)
            if self.use_2opt:
                t, f = two_opt_fast(t, self.D, max_iter=50)
            else:
                f = tour_length(t, self.D)
            h = self._hash(t)
            if h not in hashes:
                pop.append(t); fits.append(f); hashes.add(h)
            if len(pop) >= self.pop_size: break
        attempts = 0
        while len(pop) < self.pop_size and attempts < 10_000:
            t = list(range(self.n)); self.rng.shuffle(t)
            h = self._hash(t)
            if h not in hashes:
                pop.append(t); fits.append(tour_length(t, self.D)); hashes.add(h)
            attempts += 1
        return pop, fits, hashes

    def _ox(self, p1, p2):
        n = self.n; a, b = sorted(self.rng.sample(range(n), 2))
        child = [-1]*n; child[a:b+1] = p1[a:b+1]
        seg = set(p1[a:b+1]); pos = (b+1)%n
        for city in (p2[(b+1+i)%n] for i in range(n)):
            if city not in seg:
                child[pos] = city; pos = (pos+1)%n
        return child

    def _mutate(self, tour):
        if self.rng.random() < self.pm:
            t = tour[:]; a, b = sorted(self.rng.sample(range(self.n), 2))
            t[a:b+1] = t[a:b+1][::-1]; return t
        return tour

    def _tournament(self, pop, fits):
        a, b = self.rng.sample(range(len(pop)), 2)
        return pop[a] if fits[a] <= fits[b] else pop[b]

    def _try_insert(self, child, child_len, pop, fits, hashes):
        h = self._hash(child)
        if h in hashes: return False
        divs = [edge_diversity_fast(child, p) for p in pop]
        sim_idx = int(np.argmin(divs))
        if divs[sim_idx] < self.div_thresh:
            if child_len < fits[sim_idx]:
                hashes.discard(self._hash(pop[sim_idx]))
                pop[sim_idx] = child; fits[sim_idx] = child_len; hashes.add(h)
                return True
        else:
            worst = int(np.argmax(fits))
            if child_len < fits[worst]:
                hashes.discard(self._hash(pop[worst]))
                pop[worst] = child; fits[worst] = child_len; hashes.add(h)
                return True
        return False

    def run(self, budget_seconds: float):
        pop, fits, hashes = self._init_pop()
        best_len = min(fits); history = [best_len]
        t0 = time.time()
        while time.time() - t0 < budget_seconds:
            p1 = self._tournament(pop, fits); p2 = self._tournament(pop, fits)
            child = self._ox(p1, p2) if self.rng.random() < self.pc else p1[:]
            child = self._mutate(child)
            child, child_len = (two_opt_fast(child, self.D, 30) if self.use_2opt
                                else (child, tour_length(child, self.D)))
            self._try_insert(child, child_len, pop, fits, hashes)
            cur = min(fits)
            if cur < best_len: best_len = cur
            history.append(best_len)
        return best_len, history


def cbga_solver(D, n, seed, budget):
    return ChuBeasleyGA(D, n, pop_size=60, pc=0.85, pm=0.05,
                        diversity_threshold=0.2, use_2opt=True, seed=seed).run(budget)
