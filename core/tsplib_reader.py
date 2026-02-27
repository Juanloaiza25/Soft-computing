import os


def _parse_value(line: str) -> str:
    """
    Extrae el valor de una línea de cabecera TSPLIB.
    Soporta ambos formatos válidos del estándar:
        "KEY : value"   (con dos puntos)
        "KEY value"     (con espacio)
    """
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return line.split(None, 1)[1].strip()


def read_tsplib(filepath: str):
    """
    Lee un archivo TSPLIB .tsp y devuelve:
        coords    (list de tuplas (x, y))
        dimension (int)
        edge_type (str)

    Soporta separadores con y sin ':' en la cabecera.
    Valida que el número de coordenadas leídas coincida con DIMENSION.
    """
    coords     = []
    dimension  = None
    edge_type  = None

    with open(filepath, "r") as f:
        lines = f.readlines()

    reading_coords = False

    for line in lines:
        line = line.strip()

        if line.startswith("DIMENSION"):
            # FIX: _parse_value maneja "DIMENSION : 52" y "DIMENSION 52"
            dimension = int(_parse_value(line))

        elif line.startswith("EDGE_WEIGHT_TYPE"):
            # FIX: _parse_value maneja "EDGE_WEIGHT_TYPE : EUC_2D" y "EDGE_WEIGHT_TYPE EUC_2D"
            edge_type = _parse_value(line)

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

    # FIX: validar que se leyeron exactamente los nodos esperados
    if dimension is not None and len(coords) != dimension:
        raise ValueError(
            f"TSPLIB '{os.path.basename(filepath)}': "
            f"se esperaban {dimension} nodos, se leyeron {len(coords)}"
        )

    return coords, dimension, edge_type