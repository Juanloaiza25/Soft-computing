import numpy as np

def build_distance_matrix(coords, dist_type='EUC_2D'):
    xy = np.array(coords, dtype=np.float64)
    dx = xy[:, 0:1] - xy[:, 0]
    dy = xy[:, 1:2] - xy[:, 1]
    if dist_type == 'ATT':
        r = np.sqrt((dx**2 + dy**2) / 10.0)
        t = r.astype(int)
        D = np.where(t < r, t + 1, t).astype(np.float64)
    else:
        D = np.round(np.sqrt(dx**2 + dy**2))
    np.fill_diagonal(D, 0)
    return D


def tour_length(tour, D):
    t = np.asarray(tour)
    return float(D[t, np.roll(t, -1)].sum())


def gap(length, optimal):
    return (length - optimal) / optimal * 100


def nearest_neighbor(D, start):
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


def two_opt_fast(tour, D, max_iter=200):
    """
    2-opt local search con estrategia 'first improvement'.
    Tras cada swap se reinicia la búsqueda para evitar usar
    índices obsoletos (a, b quedan stale tras la reversión).
    """
    t = np.array(tour, dtype=np.int32)
    n = len(t)
    best_len = float(D[t, np.roll(t, -1)].sum())

    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 1):
            a, b = t[i - 1], t[i]
            for j in range(i + 1, n):
                c, d = t[j], t[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-10:
                    t[i:j+1] = t[i:j+1][::-1]
                    best_len += delta
                    improved = True
                    break           # ← romper loop j: a,b ya son stale
            if improved:
                break               # ← romper loop i: reiniciar pasada
        if not improved:
            break

    return t.tolist(), best_len