"""
EvoSnake — Widget del tablero (Kivy Canvas)
============================================
Dibuja el estado de un objeto SnakeGame sobre un canvas de Kivy,
con colores diferenciados para cabeza, cuerpo (gradiente) y comida.
"""

from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Ellipse, Line

from utils.constants import GRID_SIZE, CELL_SIZE, COLORS


class SnakeBoard(Widget):
    """
    Widget Kivy que renderiza el estado visual del juego Snake.

    Uso
    ---
    board = SnakeBoard(game)   # game es una instancia de SnakeGame
    board.draw()               # redibujar después de cada step()
    """

    def __init__(self, game, **kwargs):
        super().__init__(**kwargs)
        self.game = game
        self.size_hint = (1, 1)
        self.cell_size = CELL_SIZE
        self._offset_x = 0
        self._offset_y = 0
        self.bind(size=self._update_size, pos=self._update_size)

    def _update_size(self, *args):
        # El tablero debe ser cuadrado, usamos la menor dimensión
        dim = min(self.width, self.height)
        self.cell_size = dim / max(1, GRID_SIZE)
        # Centrar el dibujo en el área asignada
        self._offset_x = self.x + (self.width - dim) / 2
        self._offset_y = self.y + (self.height - dim) / 2
        self.draw()

    def draw(self):
        """Redibuja completamente el canvas con el estado actual del juego."""
        self.canvas.clear()
        g = self.game
        pad = 2  # padding interior de cada celda

        with self.canvas:
            # ── Fondo ──
            Color(*COLORS["bg"])
            dim = self.cell_size * GRID_SIZE
            Rectangle(pos=(self._offset_x, self._offset_y), size=(dim, dim))

            # ── Líneas de la cuadrícula (sutiles) ──
            Color(*COLORS["grid_line"])
            for c in range(GRID_SIZE + 1):
                Rectangle(
                    pos=(self._offset_x + c * self.cell_size, self._offset_y),
                    size=(1, dim),
                )
            for r in range(GRID_SIZE + 1):
                Rectangle(
                    pos=(self._offset_x, self._offset_y + r * self.cell_size),
                    size=(dim, 1),
                )

            # ── Resplandor de la comida (glow) ──
            if not g.dead:
                glow = COLORS["food_glow"]
                Color(*glow)
                fc, fr = g.food
                glow_pad = self.cell_size * 0.6
                Ellipse(
                    pos=(
                        self._offset_x + fc * self.cell_size - glow_pad / 2,
                        self._offset_y + fr * self.cell_size - glow_pad / 2,
                    ),
                    size=(self.cell_size + glow_pad, self.cell_size + glow_pad),
                )

            # ── Comida (círculo rojo) ──
            Color(*COLORS["food"])
            fc, fr = g.food
            food_pad = max(1, self.cell_size * 0.15)
            Ellipse(
                pos=(
                    self._offset_x + fc * self.cell_size + food_pad,
                    self._offset_y + fr * self.cell_size + food_pad,
                ),
                size=(self.cell_size - food_pad * 2, self.cell_size - food_pad * 2),
            )

            # ── Cuerpo de la serpiente (gradiente de cola) ──
            pad = max(1, self.cell_size * 0.1)
            for col, row in g.body:
                if not g.dead:
                    body = COLORS["snake_body"]
                else:
                    body = COLORS["snake_dead"]

                Color(*body)

                Rectangle(
                    pos=(
                        self._offset_x + col * self.cell_size + pad,
                        self._offset_y + row * self.cell_size + pad,
                    ),
                    size=(self.cell_size - pad * 2, self.cell_size - pad * 2),
                )

            # ── Cabeza de la serpiente ──
            hcol, hrow = g.head
            head_color = COLORS["snake_head"] if not g.dead else COLORS["snake_dead"]
            Color(*head_color)
            head_pad = max(0.5, self.cell_size * 0.05)
            Rectangle(
                pos=(
                    self._offset_x + hcol * self.cell_size + head_pad,
                    self._offset_y + hrow * self.cell_size + head_pad,
                ),
                size=(self.cell_size - head_pad * 2, self.cell_size - head_pad * 2),
            )

            # ── Ojos de la serpiente ──
            self._draw_eyes(hcol, hrow, (g.x_vel, g.y_vel))

    def _draw_eyes(self, col, row, direction):
        """Dibuja dos ojos pequeños en la cabeza según la dirección actual."""
        cx = self._offset_x + col * self.cell_size + self.cell_size / 2
        cy = self._offset_y + row * self.cell_size + self.cell_size / 2
        eye_size = max(2, self.cell_size * 0.15)
        eye_offset = self.cell_size * 0.2  # distancia del centro al ojo

        dx, dy = direction

        if dx == 1:   # RIGHT
            offsets = [(eye_offset, eye_offset), (eye_offset, -eye_offset)]
        elif dx == -1: # LEFT
            offsets = [(-eye_offset, eye_offset), (-eye_offset, -eye_offset)]
        elif dy == -1:  # UP
            offsets = [(-eye_offset, eye_offset), (eye_offset, eye_offset)]
        else:          # DOWN
            offsets = [(-eye_offset, -eye_offset), (eye_offset, -eye_offset)]

        Color(0.1, 0.1, 0.1)
        for ox, oy in offsets:
            Ellipse(
                pos=(cx + ox - eye_size / 2, cy + oy - eye_size / 2),
                size=(eye_size, eye_size),
            )