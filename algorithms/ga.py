import random
import numpy as np
import time
from core.distance import tour_length, nearest_neighbor

class GeneticAlgorithm:
    def __init__(self, D, n, pop_size=100, pc=0.85, pm=0.03, elite_k=2, seed=0):
        self.D = D; self.n = n; self.pop_size = pop_size
        self.pc = pc; self.pm = pm; self.elite_k = elite_k
        self.rng = random.Random(seed)

    def _init_pop(self):
        pop = []
        n_nn = max(1, self.pop_size // 5)
        for s in self.rng.sample(range(self.n), min(n_nn, self.n)):
            pop.append(nearest_neighbor(self.D, s))
        while len(pop) < self.pop_size:
            t = list(range(self.n)); self.rng.shuffle(t); pop.append(t)
        return pop

    def _tournament(self, pop, fits):
        a, b = self.rng.sample(range(len(pop)), 2)
        winner = a if fits[a] <= fits[b] else b
        return pop[winner][:]   # copia

    def _ox_crossover(self, p1, p2):
        n = self.n
        a, b = sorted(self.rng.sample(range(n), 2))
        child = [-1] * n; child[a:b+1] = p1[a:b+1]
        seg = set(p1[a:b+1]); pos = (b+1) % n
        for city in (p2[(b+1+i) % n] for i in range(n)):
            if city not in seg:
                child[pos] = city; pos = (pos+1) % n
        return child

    def _mutate(self, tour):
        if self.rng.random() < self.pm:
            t = tour[:]; a, b = sorted(self.rng.sample(range(self.n), 2))
            t[a:b+1] = t[a:b+1][::-1]; return t
        return tour

    def run(self, budget_seconds: float):
        t0 = time.time()
        pop = self._init_pop()
        fits = np.array([tour_length(t, self.D) for t in pop])
        best_len = float(fits.min())
        history = [best_len]

        while time.time() - t0 < budget_seconds:
            idx = np.argsort(fits)
            elite_fits = fits[idx[:self.elite_k]].tolist()      # fits conocidos
            elites = [pop[i][:] for i in idx[:self.elite_k]]
            new_pop = elites[:]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                child = self._ox_crossover(p1, p2) if self.rng.random() < self.pc else p1[:]
                child = self._mutate(child)
                new_pop.append(child)
            
            pop = new_pop
            # ✅ recalcular solo los hijos, reusar fits de élites
            child_fits = [tour_length(t, self.D) for t in pop[self.elite_k:]]
            fits = np.array(elite_fits + child_fits)
            
            g = float(fits.min())
            if g < best_len:
                best_len = g
            history.append(best_len)
            
        return best_len, history



def ga_solver(D, n, seed, budget):
    return GeneticAlgorithm(D, n, pop_size=100, pc=0.85, pm=0.03, elite_k=2, seed=seed).run(budget)