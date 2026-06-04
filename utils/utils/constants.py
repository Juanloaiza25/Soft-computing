"""
SnakeAI — Constantes (fiel al proyecto Processing original)
=============================================================
"""

# ──────────────────────────────────────────────
# GRID  (el original usa pixeles; aquí usamos celdas)
# El area de juego original: 38×38 celdas (width-400-40)/20 × (height-40)/20
# ──────────────────────────────────────────────
GRID_SIZE  = 20       # celdas por lado (tablero cuadrado)
CELL_SIZE  = 20       # píxeles por celda en la UI Kivy

# ──────────────────────────────────────────────
# RED NEURONAL
# ──────────────────────────────────────────────
NN_INPUT_SIZE   = 26   # 8 direcciones × 3 (food, body, 1/dist_wall) + food_dx + food_dy
NN_HIDDEN_NODES = 12   # neuronas por capa oculta
NN_HIDDEN_LAYERS = 2   # número de capas ocultas
NN_OUTPUT_SIZE  = 3    # 0: Adelante, 1: Izquierda, 2: Derecha

# ──────────────────────────────────────────────
# SERPIENTE
# ──────────────────────────────────────────────
INITIAL_LIFE   = 200   # movimientos iniciales antes de morir
LIFE_GAIN      = 100   # movimientos ganados al comer
MAX_LIFE       = 500   # tope de movimientos restantes

# ──────────────────────────────────────────────
# ALGORITMO GENÉTICO
# ──────────────────────────────────────────────
POP_SIZE       = 2000  # tamaño de la población
MUTATION_RATE  = 0.08  # tasa de mutación por peso
GENERATIONS    = 500

# ──────────────────────────────────────────────
# VELOCIDAD
# ──────────────────────────────────────────────
FPS_PLAY   = 15        # FPS modo humano
FPS_AI     = 100       # FPS modo IA (lo más rápido posible)

# ──────────────────────────────────────────────
# PALETA DE COLORES  (neón cyberpunk sobre fondo oscuro)
# ──────────────────────────────────────────────
COLORS = {
    # Fondos
    "bg":           (0.04, 0.04, 0.07, 1),
    "grid_line":    (0.09, 0.09, 0.13, 1),
    "panel_bg":     (0.06, 0.06, 0.10, 1),
    "border":       (0.20, 0.20, 0.28, 1),

    # Serpiente
    "snake_head":   (0.00, 1.00, 0.85, 1),    # cian neón
    "snake_body":   (1.00, 1.00, 1.00, 1),     # blanco (como en el original)
    "snake_dead":   (0.45, 0.45, 0.45, 1),     # gris al morir

    # Comida
    "food":         (1.00, 0.20, 0.20, 1),     # rojo
    "food_glow":    (1.00, 0.10, 0.10, 0.15),

    # Texto
    "text":         (0.85, 0.85, 0.90, 1),
    "text_dim":     (0.50, 0.50, 0.55, 1),
    "text_accent":  (0.00, 1.00, 0.85, 1),     # cian
    "text_score":   (1.00, 0.85, 0.00, 1),     # dorado

    # Botones
    "btn_bg":       (0.12, 0.12, 0.18, 1),
    "btn_text":     (0.88, 0.88, 0.92, 1),
    "btn_active":   (0.00, 0.45, 0.35, 1),
    "btn_danger":   (0.40, 0.10, 0.10, 1),

    # NN visualization (pesos)
    "weight_pos":   (0.20, 0.20, 1.00, 1),     # azul  (peso > 0)
    "weight_neg":   (1.00, 0.00, 0.00, 1),     # rojo  (peso < 0)
    "node_on":      (0.00, 1.00, 0.00, 1),     # verde (nodo activo)
    "node_off":     (0.90, 0.90, 0.90, 1),     # blanco (nodo inactivo)
    "node_output":  (0.00, 1.00, 0.00, 1),     # salida ganadora

    # Gráfica matplotlib
    "plot_bg":      "#0A0A12",
    "plot_line":    "#FF3040",
    "plot_axis":    "#969696",
    "plot_grid":    "#1A1A24",
    "plot_text":    "#E0E0E8",
}
