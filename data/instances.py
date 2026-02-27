import os
from core.tsplib_reader import read_tsplib

TSPLIB_PATH = os.path.join(os.path.dirname(__file__), "tsplib")

FILES = [
    "berlin52.tsp",
    "eil51.tsp",
    "att48.tsp",
    "st70.tsp"
]

OPTIMALS = {
    "berlin52": 7542,
    "eil51": 426,
    "att48": 10628,
    "st70": 675
}

INSTANCES = {}

# Check if TSPLIB_PATH exists, otherwise look for it in project root
if not os.path.exists(TSPLIB_PATH):
    TSPLIB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tsplib")

for filename in FILES:
    path = os.path.join(TSPLIB_PATH, filename)
    if os.path.exists(path):
        coords, dimension, edge_type = read_tsplib(path)
        name = filename.replace(".tsp", "")
        INSTANCES[name] = {
            "coords": coords,
            "dimension": dimension,
            "type": edge_type,
            "optimal": OPTIMALS.get(name)
        }
    else:
        print(f"Warning: File {path} not found.")

if not INSTANCES:
    # If no files found, use hardcoded data as fallback (optional, but good for testing)
    pass
