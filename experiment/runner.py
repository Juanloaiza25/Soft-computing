import time
import numpy as np
from joblib import Parallel, delayed
from core.distance import gap
from data.instances import INSTANCES
from config import CFG

def _run_one(algo_fn, algo_name, instance_name, seed, budget, distance_matrix):
    """Tarea atomica para joblib: una (algo, instancia, seed)."""
    n = distance_matrix.shape[0]
    opt = INSTANCES[instance_name]['optimal']
    t0 = time.time()
    best, history = algo_fn(distance_matrix, n, seed, budget)
    return {
        'algo': algo_name, 'instance': instance_name, 'seed': seed,
        'best': best, 'gap': gap(best, opt),
        'time': time.time() - t0, 'history': history,
    }


def run_all(algo_fn, algo_name, dist_dict, seeds=None, budget=None, n_jobs=None, verbose=True):
    """
    OPTIMIZACION: paraleliza seeds x instancias con joblib Parallel.
    """
    seeds   = seeds   or CFG['BASE_SEEDS']
    budget  = budget  or CFG['BUDGET_SECONDS']
    n_jobs  = n_jobs  or CFG['N_JOBS']
    
    tasks = [
        delayed(_run_one)(algo_fn, algo_name, inst, s, budget, dist_dict[inst])
        for inst in dist_dict for s in seeds
    ]
    
    results = Parallel(n_jobs=n_jobs, backend='loky')(tasks)
    
    if verbose:
        for inst in dist_dict:
            sub = [r for r in results if r['instance'] == inst]
            if not sub: continue
            bests = [r['best'] for r in sub]
            gaps = [r['gap'] for r in sub]
            print(f"  {algo_name:6s} | {inst:10s} | best={min(bests):.0f} "
                  f"avg={np.mean(bests):.0f} std={np.std(bests):.1f} "
                  f"GAP={np.mean(gaps):.2f}%")
    return results