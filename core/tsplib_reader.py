import os

def read_tsplib(filepath):
    """
    Lee un archivo TSPLIB .tsp y devuelve:
        coords (list de tuplas)
        dimension (int)
        edge_type (str)
    """
    coords = []
    dimension = None
    edge_type = None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    reading_coords = False

    for line in lines:
        line = line.strip()

        if line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1])

        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_type = line.split(":")[1].strip()

        elif line.startswith("NODE_COORD_SECTION"):
            reading_coords = True
            continue

        elif line.startswith("EOF"):
            break

        elif reading_coords:
            parts = line.split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))

    return coords, dimension, edge_type