"""
Taller: Comparación y Afinamiento de GA vs ACO en el TSP (TSPLIB)
=================================================================
VERSIÓN OPTIMIZADA
Mejoras aplicadas:
  1. tour_length vectorizado con NumPy (sin loop Python)
  2. two_opt con delta O(1) por swap en lugar de recalcular tour completo
  3. build_distance_matrix completamente vectorizado con broadcasting
  4. Paralelización del experimento completo con joblib
  5. ACO: depósito de feromonas con np.add.at (sin loop ciudad por ciudad)
  6. CBGA: edge_diversity con frozensets de tuplas (min,max) ~2x más rápido
  7. nearest_neighbor con np.argmin + máscara en lugar de min() con generator
  8. Configuración centralizada en dict CFG
"""

# =============================================================================
# 0) IMPORTS Y CONFIGURACIÓN CENTRAL
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import time, random, warnings, csv, os
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

CFG = dict(
    BASE_SEEDS     = list(range(30)),
    BUDGET_SECONDS = 10,
    TUNING_SEEDS   = list(range(10)),
    TUNING_INST    = 'berlin52',
    COLORS         = {'GA': '#2196F3', 'ACO': '#FF9800', 'CBGA': '#4CAF50'},
    N_JOBS         = 2,
    OUT_DIR        = '/mnt/user-data/outputs',
)
os.makedirs(CFG['OUT_DIR'], exist_ok=True)
print("Imports listos. Seeds:", CFG['BASE_SEEDS'][:5], "... (30 en total)")
print(f"  Budget: {CFG['BUDGET_SECONDS']}s  |  Jobs paralelos: {CFG['N_JOBS']}")


# =============================================================================
# 1) INSTANCIAS TSPLIB EMBEBIDAS
# =============================================================================
BERLIN52_COORDS = [
    (565,575),(25,185),(345,750),(945,685),(845,655),(880,660),(25,230),
    (525,1000),(580,1175),(650,1130),(1605,620),(1220,580),(1465,200),
    (1530,5),(845,680),(725,370),(145,665),(415,635),(510,875),(560,365),
    (300,465),(520,585),(480,415),(835,625),(975,580),(1215,245),(1320,315),
    (1250,400),(660,180),(410,250),(420,555),(575,665),(1150,1160),(700,580),
    (685,595),(685,610),(770,610),(795,645),(720,635),(760,650),(475,960),
    (95,260),(875,920),(700,500),(555,815),(830,485),(1170,65),(830,610),
    (605,625),(595,360),(1340,725),(1740,245)
]
EIL51_COORDS = [
    (37,52),(49,49),(52,64),(20,26),(40,30),(21,47),(17,63),(31,62),(52,33),
    (51,21),(42,41),(31,32),(5,25),(12,42),(36,16),(52,41),(27,23),(17,33),
    (13,13),(57,58),(62,42),(42,57),(16,57),(8,52),(7,38),(27,68),(30,48),
    (43,67),(58,48),(58,27),(37,69),(38,46),(46,10),(61,33),(62,63),(63,69),
    (32,22),(45,35),(59,15),(5,6),(10,17),(21,10),(5,64),(30,15),(39,10),
    (32,39),(25,32),(25,55),(48,28),(56,37),(30,40)
]
ATT48_COORDS = [
    (6734,1453),(2233,10),(5530,1424),(401,841),(3082,1644),(7608,4458),
    (7573,3716),(7265,1268),(6898,1885),(1112,2049),(5468,2606),(5989,2873),
    (4706,2674),(4612,2035),(6347,2683),(6107,669),(7611,5184),(7462,3590),
    (7732,4723),(5900,3561),(4483,3369),(6101,1110),(5199,2182),(1633,2809),
    (4307,2322),(675,1006),(7555,4819),(7541,3981),(3177,756),(7352,4506),
    (7545,2801),(3245,3305),(6426,3173),(4608,1198),(23,2216),(7248,3779),
    (7762,4595),(7392,2244),(3484,2829),(6271,2135),(4985,140),(1916,1569),
    (7280,4899),(7509,3239),(10,2676),(6807,2993),(5185,3258),(3023,1942)
]
ST70_COORDS = [
    (64,96),(80,39),(69,23),(72,42),(48,67),(58,43),(81,34),(79,17),
    (30,23),(42,67),(7,76),(29,51),(78,92),(64,8),(95,57),(14,3),
    (72,77),(87,63),(89,55),(57,91),(51,8),(41,95),(55,21),(40,88),
    (28,58),(45,18),(75,82),(61,77),(74,56),(68,33),(43,28),(71,51),
    (86,69),(67,87),(31,35),(25,72),(16,39),(39,94),(38,29),(31,93),
    (77,42),(77,65),(17,22),(8,89),(11,54),(49,53),(63,50),(40,5),
    (46,40),(95,38),(58,71),(63,21),(90,21),(28,71),(82,58),(21,12),
    (95,23),(82,23),(22,53),(65,37),(57,72),(3,85),(70,11),(71,85),
    (67,4),(54,52),(82,49),(72,6),(56,39),(84,81),(37,52),(52,43)
]

INSTANCES = {
    'berlin52': {'coords': BERLIN52_COORDS, 'optimal': 7542,  'type': 'EUC_2D'},
    'eil51':    {'coords': EIL51_COORDS,    'optimal': 426,   'type': 'EUC_2D'},
    'att48':    {'coords': ATT48_COORDS,    'optimal': 10628, 'type': 'ATT'},
    'st70':     {'coords': ST70_COORDS,     'optimal': 675,   'type': 'EUC_2D'},
}


# =============================================================================
# 2) FUNCIONES CORE OPTIMIZADAS
# =============================================================================

def build_distance_matrix(coords: list, dist_type: str = 'EUC_2D') -> np.ndarray:
    """
    OPTIMIZACION: broadcasting NumPy puro, sin doble loop Python.
    Original: O(n^2) iteraciones Python.
    Ahora:    una operacion vectorial de forma (n,n).
    """
    xy = np.array(coords, dtype=np.float64)
    dx = xy[:, 0:1] - xy[:, 0]   # (n,n)
    dy = xy[:, 1:2] - xy[:, 1]   # (n,n)
    if dist_type == 'ATT':
        r = np.sqrt((dx**2 + dy**2) / 10.0)
        t = r.astype(int)
        D = np.where(t < r, t + 1, t).astype(np.float64)
    else:
        D = np.round(np.sqrt(dx**2 + dy**2))
    np.fill_diagonal(D, 0)
    return D


def tour_length(tour, D: np.ndarray) -> float:
    """
    OPTIMIZACION: indexing vectorial en lugar de sum() con loop Python.
    Speedup tipico: 5-10x.
    """
    t = np.asarray(tour)
    return float(D[t, np.roll(t, -1)].sum())


def gap(length: float, optimal: int) -> float:
    return (length - optimal) / optimal * 100


def nearest_neighbor(D: np.ndarray, start: int) -> list:
    """
    OPTIMIZACION: np.argmin con mascara de infinito en lugar
    de min() con generator Python.
    """
    n = D.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        row = D[tour[-1]].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        tour.append(nxt)
        visited[nxt] = True
    return tour


def two_opt_fast(tour: list, D: np.ndarray, max_iter: int = 200) -> tuple:
    """
    OPTIMIZACION CLAVE: calculo de delta O(1) por swap.
    Original: recalculaba tour_length completo O(n) en cada intento.
    Ahora:    solo consulta 4 celdas de D para decidir si el swap mejora.

    Formula:
      delta = D[t[i-1], t[j]] + D[t[i], t[j+1]]
            - D[t[i-1], t[i]] - D[t[j], t[j+1]]
      Si delta < 0 -> el swap es beneficioso.
    """
    t = np.array(tour, dtype=np.int32)
    n = len(t)
    best_len = float(D[t, np.roll(t, -1)].sum())

    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = t[i - 1], t[i]
                c, d = t[j], t[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-10:
                    t[i:j + 1] = t[i:j + 1][::-1]
                    best_len += delta
                    improved = True
                    break           # FIX: romper loop j — a,b ya son stale
            if improved:
                break               # FIX: romper loop i — reiniciar pasada
        if not improved:
            break
    return t.tolist(), best_len


# =============================================================================
# 3) ALGORITMO GENETICO CLASICO (GA)
# =============================================================================

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

    # FIX CRITICO: devolver copia [:] para evitar aliasing con pop
    def _tournament(self, pop, fits):
        a, b = self.rng.sample(range(len(pop)), 2)
        winner = a if fits[a] <= fits[b] else b
        return pop[winner][:]

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
        # FIX CRITICO: t0 ANTES de _init_pop — el tiempo de inicializacion
        # cuenta como parte del presupuesto
        t0 = time.time()
        pop = self._init_pop()
        fits = np.array([tour_length(t, self.D) for t in pop])
        best_len = float(fits.min())
        history = [best_len]

        while time.time() - t0 < budget_seconds:
            idx = np.argsort(fits)
            # FIX MENOR: preservar elite_fits para no recalcular tours ya conocidos
            elite_fits = fits[idx[:self.elite_k]].tolist()
            elites = [pop[i][:] for i in idx[:self.elite_k]]
            new_pop = elites[:]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                child = self._ox_crossover(p1, p2) if self.rng.random() < self.pc else p1[:]
                child = self._mutate(child)
                new_pop.append(child)
            pop = new_pop
            # FIX MENOR: recalcular solo hijos, reusar fits de élites
            child_fits = [tour_length(t, self.D) for t in pop[self.elite_k:]]
            fits = np.array(elite_fits + child_fits)
            g = float(fits.min())
            if g < best_len: best_len = g
            history.append(best_len)
        return best_len, history


def ga_solver(D, n, seed, budget):
    return GeneticAlgorithm(D, n, pop_size=100, pc=0.85, pm=0.03, elite_k=2, seed=seed).run(budget)


# =============================================================================
# 4) ACO — OPTIMIZADO
# =============================================================================

class AntColonyOptimization:
    """
    Mejoras vs original:
      - Pre-calculo de eta^beta en __init__ (constante, no cambia nunca)
      - Deposito de feromonas con np.add.at (sin loop ciudad por ciudad)
      - Limites tau_min / tau_max para estabilidad numerica
      - Evita tau**alpha cuando alpha==1.0
    """
    def __init__(self, D, n, n_ants=20, alpha=1.0, beta=5.0, rho=0.3, Q=100.0, seed=0):
        self.D = D; self.n = n; self.n_ants = n_ants
        self.alpha = alpha; self.beta = beta; self.rho = rho; self.Q = Q
        self.rng = np.random.default_rng(seed)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.eta = np.where(D > 0, 1.0 / D, 0.0)
        # FIX CRITICO: precalcular eta_b UNA sola vez — es constante
        self.eta_b = self.eta ** self.beta
        tau0 = 1.0 / (n * float(np.mean(D[D > 0])))
        self.tau = np.full((n, n), tau0, dtype=np.float64)
        self.tau_min, self.tau_max = 1e-10, 1e6

    def _construct(self) -> list:
        start = int(self.rng.integers(0, self.n))
        visited = np.zeros(self.n, dtype=bool)
        tour = [start]; visited[start] = True
        for _ in range(self.n - 1):
            cur = tour[-1]
            # FIX MENOR: evitar tau**alpha cuando alpha==1.0
            tau_row = self.tau[cur] if self.alpha == 1.0 else self.tau[cur] ** self.alpha
            desire = tau_row * self.eta_b[cur]
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
            # FIX CRITICO: usa self.eta_b precalculado en __init__
            tours = [self._construct() for _ in range(self.n_ants)]
            lengths = [tour_length(t, self.D) for t in tours]
            it_best = min(lengths)
            if it_best < best_len: best_len = it_best
            self._update(tours, lengths)
            history.append(best_len)
        return best_len, history


def aco_solver(D, n, seed, budget):
    return AntColonyOptimization(D, n, n_ants=20, alpha=1.0, beta=5.0,
                                 rho=0.3, Q=100.0, seed=seed).run(budget)


# =============================================================================
# 5) CBGA — OPTIMIZADO
# =============================================================================

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

    # FIX CRITICO: devolver copia [:] para evitar aliasing con pop
    def _tournament(self, pop, fits):
        a, b = self.rng.sample(range(len(pop)), 2)
        winner = a if fits[a] <= fits[b] else b
        return pop[winner][:]

    def _try_insert(self, child, child_len, pop, fits, hashes):
        h = self._hash(child)
        if h in hashes: return False
        divs = [edge_diversity_fast(child, p) for p in pop]
        sim_idx = int(np.argmin(divs))
        if divs[sim_idx] < self.div_thresh:
            if child_len < fits[sim_idx]:
                # FIX MENOR: calcular hash del desplazado UNA sola vez
                displaced_hash = self._hash(pop[sim_idx])
                hashes.discard(displaced_hash)
                pop[sim_idx] = child; fits[sim_idx] = child_len; hashes.add(h)
                return True
        else:
            worst = int(np.argmax(fits))
            if child_len < fits[worst]:
                displaced_hash = self._hash(pop[worst])
                hashes.discard(displaced_hash)
                pop[worst] = child; fits[worst] = child_len; hashes.add(h)
                return True
        return False

    def run(self, budget_seconds: float):
        # FIX CRITICO: t0 ANTES de _init_pop — el tiempo de inicializacion
        # (hasta ~530ms en st70) cuenta como parte del presupuesto
        t0 = time.time()
        pop, fits, hashes = self._init_pop()
        best_len = min(fits); history = [best_len]
        while time.time() - t0 < budget_seconds:
            p1 = self._tournament(pop, fits); p2 = self._tournament(pop, fits)
            child = self._ox(p1, p2) if self.rng.random() < self.pc else p1[:]
            child = self._mutate(child)
            child, child_len = (two_opt_fast(child, self.D, 30) if self.use_2opt
                                else (child, tour_length(child, self.D)))
            # FIX MENOR: comparacion directa en vez de min(fits) O(n)
            inserted = self._try_insert(child, child_len, pop, fits, hashes)
            if inserted and child_len < best_len:
                best_len = child_len
            history.append(best_len)
        return best_len, history


def cbga_solver(D, n, seed, budget):
    return ChuBeasleyGA(D, n, pop_size=60, pc=0.85, pm=0.05,
                        diversity_threshold=0.2, use_2opt=True, seed=seed).run(budget)


# =============================================================================
# 6) RUNNER EXPERIMENTAL PARALELIZADO
# =============================================================================

def _run_one(algo_fn, algo_name, instance_name, seed, budget):
    """Tarea atomica para joblib: una (algo, instancia, seed)."""
    D = DIST[instance_name]
    n = D.shape[0]
    opt = INSTANCES[instance_name]['optimal']
    t0 = time.time()
    best, history = algo_fn(D, n, seed, budget)
    return {
        'algo': algo_name, 'instance': instance_name, 'seed': seed,
        'best': best, 'gap': gap(best, opt),
        'time': time.time() - t0, 'history': history,
    }


def run_all(algo_fn, algo_name, seeds=None, budget=None, n_jobs=None, verbose=True):
    """
    OPTIMIZACION: paraleliza seeds x instancias con joblib Parallel.
    Con n_jobs=2, el tiempo total se reduce ~a la mitad.
    """
    # FIX CRITICO: 'is None' en vez de 'or' — para que n_jobs=0 no sea ignorado
    seeds   = CFG['BASE_SEEDS']     if seeds  is None else seeds
    budget  = CFG['BUDGET_SECONDS'] if budget is None else budget
    n_jobs  = CFG['N_JOBS']         if n_jobs is None else n_jobs
    tasks = [
        delayed(_run_one)(algo_fn, algo_name, inst, s, budget)
        for inst in INSTANCES for s in seeds
    ]
    results = Parallel(n_jobs=n_jobs, backend='loky')(tasks)
    if verbose:
        for inst in INSTANCES:
            sub = [r for r in results if r['instance'] == inst]
            bests = [r['best'] for r in sub]
            print(f"  {algo_name:6s} | {inst:10s} | best={min(bests):.0f} "
                  f"avg={np.mean(bests):.0f} std={np.std(bests):.1f} "
                  f"GAP={np.mean([r['gap'] for r in sub]):.2f}%")
    return results


# =============================================================================
# 7) EJECUCION PRINCIPAL
# =============================================================================
if __name__ == '__main__':

    # Construir matrices vectorizadas
    DIST = {}
    for name, inst in INSTANCES.items():
        DIST[name] = build_distance_matrix(inst['coords'], inst['type'])
        print(f"  {name:10s}  nodos={len(inst['coords']):3d}  optimo={inst['optimal']:6d}")
    print('\nMatrices listas.\n')

    # Prueba rapida de los tres algoritmos
    # FIX MENOR: n=D.shape[0] en vez de 52 hardcodeado
    for fn, label in [(ga_solver,'GA'),(aco_solver,'ACO'),(cbga_solver,'CBGA')]:
        D_b52 = DIST['berlin52']
        b, _ = fn(D_b52, D_b52.shape[0], seed=0, budget=3)
        print(f"  Prueba {label} berlin52 (3s, seed=0): {b:.0f}  GAP={gap(b,7542):.2f}%")
    print()

    # Experimento completo paralelizado
    all_results = []
    for fn, name in [(ga_solver,'GA'),(aco_solver,'ACO'),(cbga_solver,'CBGA')]:
        icon = {'GA':'[GA]','ACO':'[ACO]','CBGA':'[CBGA]'}[name]
        print(f"{'='*60}\n{icon} Ejecutando {name}...\n{'='*60}")
        all_results.extend(run_all(fn, name))
    print('\nExperimento completo.\n')

    # --- Tabla resumen ---
    ALGOS = ['GA', 'ACO', 'CBGA']
    print('TABLA RESUMEN')
    print('='*90)
    hdr = (f"{'Inst':10s} {'Algo':6s} {'Optimo':7s} {'Best':7s} {'Avg':8s} "
           f"{'Std':7s} {'Peor':7s} {'GAP_b%':7s} {'GAP_a%':7s} {'T(s)':6s}")
    print(hdr); print('-'*len(hdr))
    prev = None
    for inst in INSTANCES:
        opt = INSTANCES[inst]['optimal']
        for algo in ALGOS:
            sub   = [r for r in all_results if r['instance']==inst and r['algo']==algo]
            bests = [r['best'] for r in sub]
            if inst != prev and prev is not None: print()
            prev = inst
            print(f"{inst:10s} {algo:6s} {opt:7d} {int(min(bests)):7d} "
                  f"{np.mean(bests):8.1f} {np.std(bests):7.1f} {int(max(bests)):7d} "
                  f"{gap(min(bests),opt):7.2f} {np.mean([gap(b,opt) for b in bests]):7.2f} "
                  f"{np.mean([r['time'] for r in sub]):6.2f}")

    # --- Guardar CSV ---
    csv_path = os.path.join(CFG['OUT_DIR'], 'resultados_optimizado.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['algo','instance','seed','best','gap','time'])
        w.writeheader()
        for r in all_results:
            w.writerow({k: r[k] for k in ['algo','instance','seed','best','gap','time']})
    print(f'\nCSV guardado: {csv_path}  ({len(all_results)} filas)')

    # --- Figura 1: Boxplots ---
    COLORS = CFG['COLORS']
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle('Distribucion de Mejores Distancias (30 seeds)', fontsize=15, fontweight='bold')
    for ax, inst in zip(axes, INSTANCES):
        opt = INSTANCES[inst]['optimal']
        data = [[r['best'] for r in all_results if r['instance']==inst and r['algo']==a] for a in ALGOS]
        bp = ax.boxplot(data, tick_labels=ALGOS, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        for patch, a in zip(bp['boxes'], ALGOS):
            patch.set_facecolor(COLORS[a]); patch.set_alpha(0.75)
        ax.axhline(opt, color='red', linestyle='--', linewidth=1.5, label=f'Optimo ({opt})')
        ax.set_title(f'{inst}\n(optimo = {opt})', fontsize=11)
        ax.set_ylabel('Longitud del tour'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['OUT_DIR'], 'fig1_boxplots.png'), dpi=130, bbox_inches='tight')
    plt.close(); print('fig1_boxplots.png guardada')

    # --- Figura 2: Convergencia ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Curvas de Convergencia Promedio', fontsize=14, fontweight='bold')
    for ax, inst in zip(axes.flatten(), INSTANCES):
        opt = INSTANCES[inst]['optimal']
        for algo in ALGOS:
            sub = [r for r in all_results if r['instance']==inst and r['algo']==algo]
            ml = min(len(r['history']) for r in sub)
            mat = np.array([r['history'][:ml] for r in sub])
            mh, sh = mat.mean(0), mat.std(0)
            ax.plot(mh, color=COLORS[algo], label=algo, linewidth=2)
            ax.fill_between(range(ml), mh-sh, mh+sh, color=COLORS[algo], alpha=0.15)
        ax.axhline(opt, color='red', linestyle='--', linewidth=1.5, label='Optimo')
        ax.set_title(inst, fontsize=12); ax.set_xlabel('Iteracion'); ax.set_ylabel('best-so-far')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['OUT_DIR'], 'fig2_convergence.png'), dpi=130, bbox_inches='tight')
    plt.close(); print('fig2_convergence.png guardada')

    # --- Figura 3: GAP barras ---
    # FIX CRITICO: etiquetas y ylim para GAP negativos
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(INSTANCES)); width = 0.25
    for i, algo in enumerate(ALGOS):
        gv = [np.mean([r['gap'] for r in all_results if r['instance']==inst and r['algo']==algo])
              for inst in INSTANCES]
        bars = ax.bar(x + i*width, gv, width, label=algo, color=COLORS[algo],
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, gv):
            # FIX CRITICO: va='bottom' + height+0.1 rota para GAP negativos
            ypos = val + 0.1 if val >= 0 else val - 0.1
            va   = 'bottom'  if val >= 0 else 'top'
            ax.text(bar.get_x()+bar.get_width()/2, ypos,
                    f'{val:.1f}%', ha='center', va=va, fontsize=8.5, fontweight='bold')
    ax.set_xticks(x+width); ax.set_xticklabels(list(INSTANCES.keys()), fontsize=11)
    ax.set_ylabel('GAP promedio (%)'); ax.set_title('GAP promedio respecto al optimo (30 seeds)',
                                                     fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.4); ax.axhline(0, color='black', linewidth=0.7)
    # FIX CRITICO: ylim automatico para barras negativas
    all_gaps = [np.mean([r['gap'] for r in all_results if r['instance']==inst and r['algo']==algo])
                for algo in ALGOS for inst in INSTANCES]
    margin = (max(all_gaps) - min(all_gaps)) * 0.15
    ax.set_ylim(min(all_gaps) - margin, max(all_gaps) + margin)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['OUT_DIR'], 'fig3_gap_bars.png'), dpi=130, bbox_inches='tight')
    plt.close(); print('fig3_gap_bars.png guardada')

    print('\nTodo listo.')
