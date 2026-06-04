"""
EvoSnake — Visualización de Red Neuronal (Kivy)
=================================================
Widget que dibuja la arquitectura de la NN en tiempo real,
mostrando nodos, conexiones con pesos (rojo/azul) y
activaciones (verde = activo, blanco = inactivo).
"""

import numpy as np
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line, Ellipse, Rectangle
from kivy.uix.label import Label
from kivy.metrics import dp


# Etiquetas para los nodos de salida
OUTPUT_LABELS = ["ADELANTE", "IZQ", "DER"]


class NNVisualization(Widget):
    """
    Widget Kivy que renderiza en tiempo real la red neuronal,
    sus pesos (líneas rojo/azul) y activaciones (nodos verde/blanco).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nn = None
        self._layers = []       # [26, 12, 12, 3]
        self._positions = {}    # layer_idx -> [(x, y), ...]
        self._dirty = True

        self.bind(pos=self._mark_dirty, size=self._mark_dirty)

    def _mark_dirty(self, *args):
        self._dirty = True

    def set_nn(self, nn):
        """Asigna la red neuronal a visualizar."""
        self._nn = nn
        self._layers = [nn.i_nodes]
        for _ in range(nn.h_layers):
            self._layers.append(nn.h_nodes)
        self._layers.append(nn.o_nodes)
        self._dirty = True

    # ── Cálculo de posiciones ────────────────────────────────────────

    def _compute_positions(self):
        """Calcula la posición (x, y) de cada nodo en el widget."""
        if not self._layers:
            return

        n_layers = len(self._layers)
        margin_x = dp(15)
        margin_y = dp(10)

        usable_w = self.width - 2 * margin_x
        usable_h = self.height - 2 * margin_y

        self._positions = {}
        for li, n_nodes in enumerate(self._layers):
            if n_layers > 1:
                x = self.x + margin_x + (usable_w / (n_layers - 1)) * li
            else:
                x = self.x + self.width / 2

            positions = []
            for ni in range(n_nodes):
                y = self.y + margin_y + (usable_h / (n_nodes + 1)) * (ni + 1)
                positions.append((x, y))
            self._positions[li] = positions

        self._dirty = False

    # ── Dibujo ───────────────────────────────────────────────────────

    def draw(self):
        """Redibuja la red neuronal completa."""
        self.canvas.clear()

        if self._nn is None:
            with self.canvas:
                Color(0.06, 0.06, 0.10, 1)
                Rectangle(pos=self.pos, size=self.size)
            return

        if self._dirty:
            self._compute_positions()

        activations = getattr(self._nn, 'activations', None)
        node_r = dp(3.5)

        with self.canvas:
            # Fondo negro puro
            Color(0, 0, 0, 1)
            Rectangle(pos=self.pos, size=self.size)

            # ── Conexiones (líneas) ──────────────────────────
            for li in range(len(self._layers) - 1):
                w_matrix = self._nn.weights[li]
                src_pos = self._positions[li]
                dst_pos = self._positions[li + 1]
                n_src = self._layers[li]  # sin bias

                for di, (dx, dy) in enumerate(dst_pos):
                    for si in range(n_src):
                        sx, sy = src_pos[si]
                        w = float(w_matrix[di, si])
                        alpha = min(abs(w) * 0.5, 0.6)
                        if alpha < 0.05:
                            continue
                        if w > 0:
                            Color(0.15, 0.15, 1.0, alpha)
                        else:
                            Color(1.0, 0.0, 0.0, alpha)
                        Line(points=[sx, sy, dx, dy], width=1)

            # ── Nodos ────────────────────────────────────────
            for li in range(len(self._layers)):
                is_output = (li == len(self._layers) - 1)
                positions = self._positions[li]

                # Determinar nodo ganador en la capa de salida
                winner = -1
                if is_output and activations and len(activations) > li:
                    winner = int(np.argmax(activations[li]))

                for ni, (nx, ny) in enumerate(positions):
                    is_active = False
                    if activations and li < len(activations):
                        vals = activations[li]
                        if ni < len(vals):
                            val = vals[ni]
                            if is_output:
                                is_active = (ni == winner)
                            else:
                                is_active = (val > 0.1)

                    if is_active:
                        Color(0.8, 0.1, 0.1, 1)  # Rojo activo
                    else:
                        Color(0.1, 0.2, 0.8, 1)  # Azul inactivo

                    Ellipse(
                        pos=(nx - node_r, ny - node_r),
                        size=(node_r * 2, node_r * 2),
                    )

            # ── Etiquetas de salida ──────────────────────────
            if len(self._layers) > 0:
                out_positions = self._positions[len(self._layers) - 1]
                for ni, (nx, ny) in enumerate(out_positions):
                    is_winner = False
                    if activations and len(activations) > len(self._layers) - 1:
                        is_winner = (ni == int(np.argmax(
                            activations[len(self._layers) - 1]
                        )))

                    # Remover el rectángulo de fondo de las etiquetas (estilo más limpio)

        # Las etiquetas de texto se manejan como Label widgets overlay
        self._update_output_labels()

    def _update_output_labels(self):
        """Actualiza/crea labels de texto para las salidas."""
        # Eliminar labels anteriores
        for child in list(self.children):
            if hasattr(child, '_nn_label'):
                self.remove_widget(child)

        if self._nn is None or not self._positions:
            return

        activations = getattr(self._nn, 'activations', None)
        out_li = len(self._layers) - 1
        out_positions = self._positions.get(out_li, [])
        node_r = dp(3.5)

        winner = -1
        if activations and len(activations) > out_li:
            winner = int(np.argmax(activations[out_li]))

        for ni, (nx, ny) in enumerate(out_positions):
            lbl_text = OUTPUT_LABELS[ni] if ni < len(OUTPUT_LABELS) else ""
            is_winner = (ni == winner)

            lbl = Label(
                text=lbl_text,
                font_size=dp(8),
                bold=True,
                color=(1, 1, 1, 1) if is_winner else (0.6, 0.6, 0.6, 1),
                size_hint=(None, None),
                size=(dp(55), dp(12)),
                pos=(nx + node_r + dp(6), ny - dp(6)),
                halign="left",
                valign="middle",
            )
            lbl.text_size = lbl.size
            lbl._nn_label = True
            self.add_widget(lbl)
