"""
EvoSnake — Pantalla de menú principal (Kivy)
=============================================
Menú principal con fondo degradado púrpura oscuro,
título estilizado y tres botones de acción:
  · Modo Manual
  · Modo IA
  · Gráficos
"""

import os

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.metrics import dp


class GradientButton(Button):
    """Botón con esquinas redondeadas y color sólido vibrante."""

    def __init__(self, bg_color=(0.1, 0.7, 0.4, 1), **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)  # transparente
        self.background_normal = ''
        self.background_down = ''
        self._bg_color = bg_color
        self.color = (1, 1, 1, 1)
        self.bold = True
        self.font_size = dp(16)

        self.bind(pos=self._update_canvas, size=self._update_canvas)

    def _update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self._bg_color)
            RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(12)],
            )


class MenuScreen(FloatLayout):
    """
    Pantalla de menú principal de EvoSnake.
    Emite callbacks al presionar cada botón.
    """

    def __init__(self, on_manual=None, on_ai=None, on_graphs=None, **kwargs):
        super().__init__(**kwargs)
        self._on_manual = on_manual
        self._on_ai = on_ai
        self._on_graphs = on_graphs

        # Fondo degradado (se simula con dos rectángulos)
        self.bind(pos=self._draw_bg, size=self._draw_bg)

        # ── Contenedor central ──
        center_box = BoxLayout(
            orientation="vertical",
            spacing=dp(14),
            size_hint=(0.85, None),
            height=dp(370),
            pos_hint={"center_x": 0.5, "center_y": 0.55},
        )

        # Título
        title = Label(
            text="EVOSNAKE",
            font_size=dp(42),
            bold=True,
            color=(0.0, 0.95, 0.65, 1),  # verde neón / cian
            size_hint=(1, None),
            height=dp(70),
            halign="center",
            valign="middle",
        )
        title.bind(size=title.setter("text_size"))

        # Subtítulo
        subtitle = Label(
            text="Evoluciona y domina el tablero",
            font_size=dp(14),
            color=(0.7, 0.7, 0.8, 0.8),
            size_hint=(1, None),
            height=dp(30),
            halign="center",
            valign="middle",
        )
        subtitle.bind(size=subtitle.setter("text_size"))

        # Espaciador
        spacer = Label(size_hint=(1, None), height=dp(20))

        # ── Botones ──
        btn_manual = GradientButton(
            text="Modo Manual",
            bg_color=(0.12, 0.75, 0.45, 1),  # verde esmeralda
            size_hint=(1, None),
            height=dp(52),
        )
        btn_manual.bind(on_press=lambda _: self._fire_manual())

        btn_ai = GradientButton(
            text="Modo IA",
            bg_color=(0.15, 0.45, 0.85, 1),  # azul vibrante
            size_hint=(1, None),
            height=dp(52),
        )
        btn_ai.bind(on_press=lambda _: self._fire_ai())

        btn_graphs = GradientButton(
            text="Gráficos",
            bg_color=(0.75, 0.15, 0.55, 1),  # rosa/magenta
            size_hint=(1, None),
            height=dp(52),
        )
        btn_graphs.bind(on_press=lambda _: self._fire_graphs())

        # Versión
        version = Label(
            text="v1.0.0",
            font_size=dp(11),
            color=(0.45, 0.45, 0.55, 0.6),
            size_hint=(1, None),
            height=dp(30),
            halign="center",
            valign="middle",
        )
        version.bind(size=version.setter("text_size"))

        center_box.add_widget(title)
        center_box.add_widget(subtitle)
        center_box.add_widget(spacer)
        center_box.add_widget(btn_manual)
        center_box.add_widget(btn_ai)
        center_box.add_widget(btn_graphs)
        center_box.add_widget(version)

        self.add_widget(center_box)

    # ── Fondo degradado ──────────────────────────────────────────────────

    def _draw_bg(self, *args):
        self.canvas.before.clear()
        w, h = self.size
        with self.canvas.before:
            # Capa inferior: púrpura oscuro
            Color(0.12, 0.08, 0.22, 1)
            Rectangle(pos=self.pos, size=self.size)

            # Capa superior: gradiente más claro (sutil)
            Color(0.18, 0.12, 0.32, 0.6)
            Rectangle(
                pos=(self.x, self.y + h * 0.3),
                size=(w, h * 0.7),
            )

            # Resplandor superior tenue
            Color(0.25, 0.18, 0.42, 0.3)
            Rectangle(
                pos=(self.x, self.y + h * 0.6),
                size=(w, h * 0.4),
            )

    # ── Callbacks ────────────────────────────────────────────────────────

    def _fire_manual(self):
        if self._on_manual:
            self._on_manual()

    def _fire_ai(self):
        if self._on_ai:
            self._on_ai()

    def _fire_graphs(self):
        if self._on_graphs:
            self._on_graphs()
