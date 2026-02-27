import os
from core.tsplib_reader import read_tsplib


OPTIMALS = {
    "berlin52": 7542,
    "eil51":    426,
    "att48":    10628,
    "st70":     675,
}

FILES = [f"{name}.tsp" for name in OPTIMALS]

# FIX 1: buscar TSPLIB_PATH en dos ubicaciones candidatas y fallar rápido
_here = os.path.dirname(__file__)
_candidates = [
    os.path.join(_here, "tsplib"),
    os.path.join(os.path.dirname(_here), "tsplib"),
]
TSPLIB_PATH = next((p for p in _candidates if os.path.isdir(p)), None)

if TSPLIB_PATH is None:
    raise FileNotFoundError(
        "Directorio 'tsplib' no encontrado. Rutas buscadas:\n"
        + "\n".join(f"  {p}" for p in _candidates)
    )

# Cargar instancias
INSTANCES = {}

for filename in FILES:
    path = os.path.join(TSPLIB_PATH, filename)

    if not os.path.exists(path):
        print(f"Warning: archivo no encontrado: {path}")
        continue

    coords, dimension, edge_type = read_tsplib(path)

    # FIX 5: os.path.splitext es robusto ante extensiones inesperadas
    name = os.path.splitext(filename)[0]

    INSTANCES[name] = {
        "coords":    coords,
        "dimension": dimension,
        # FIX 3: valor por defecto explícito si edge_type no está en el archivo
        "type":      edge_type or "EUC_2D",
        # FIX 2: KeyError inmediato si el nombre no está en OPTIMALS
        "optimal":   OPTIMALS[name],
    }

# FIX 4: fallar con mensaje claro en lugar de código muerto
if not INSTANCES:
    raise RuntimeError(
        "No se cargó ninguna instancia TSPLIB. "
        f"Verifica que el directorio '{TSPLIB_PATH}' contenga los archivos: "
        + ", ".join(FILES)
    )