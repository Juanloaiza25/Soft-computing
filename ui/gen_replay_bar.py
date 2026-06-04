"""
GenReplayBar — Barra de control para replay por generación
===========================================================
Slider + label que permite navegar entre generaciones guardadas.
Se integra en SnakeScreen como widget inferior al panel derecho.
"""

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.metrics import dp


class GenReplayBar(BoxLayout):
    """
    Barra con slider de generaciones + botón play/pause autoplay.
    
    Uso:
        bar = GenReplayBar(on_gen_select=callback)
        # callback recibe: gen_index (int)
    """

    def __init__(self, on_gen_select=None, **kwargs):
        super().__init__(
            orientation="vertical",
            size_hint=(1, None),
            height=dp(70),
            spacing=dp(2),
            padding=[dp(6), dp(4)],
            **kwargs
        )
        self._on_gen_select = on_gen_select
        self._total_gens = 0
        self._autoplay = False
        self._autoplay_event = None
        self._autoplay_speed = 0.4

        # ── Fila superior: info + botones ──────────────────
        top_row = BoxLayout(
            size_hint=(1, None),
            height=dp(26),
            spacing=dp(8),
        )

        self.lbl_info = Label(
            text="GEN -- / --   SCORE --   FITNESS --",
            font_size=dp(11),
            bold=True,
            color=(0.6, 0.7, 0.9, 1),
            halign="left",
            valign="middle",
            size_hint=(1, 1),
        )
        self.lbl_info.bind(size=self.lbl_info.setter("text_size"))

        self.btn_play = Button(
            text="▶  Auto",
            font_size=dp(11),
            size_hint=(None, 1),
            width=dp(72),
            background_color=(0.1, 0.2, 0.35, 1),
            bold=True,
        )
        self.btn_play.bind(on_press=lambda _: self._toggle_autoplay())

        self.btn_first = Button(
            text="|◀",
            font_size=dp(11),
            size_hint=(None, 1),
            width=dp(36),
            background_color=(0.1, 0.1, 0.18, 1),
        )
        self.btn_first.bind(on_press=lambda _: self._jump(0))

        self.btn_last = Button(
            text="▶|",
            font_size=dp(11),
            size_hint=(None, 1),
            width=dp(36),
            background_color=(0.1, 0.1, 0.18, 1),
        )
        self.btn_last.bind(on_press=lambda _: self._jump(self._total_gens - 1))

        top_row.add_widget(self.btn_first)
        top_row.add_widget(self.lbl_info)
        top_row.add_widget(self.btn_play)
        top_row.add_widget(self.btn_last)

        # ── Slider ─────────────────────────────────────────
        self.slider = Slider(
            min=0, max=1, value=0, step=1,
            size_hint=(1, None),
            height=dp(32),
            cursor_size=(dp(18), dp(18)),
        )
        self.slider.bind(value=self._on_slider)

        self.add_widget(top_row)
        self.add_widget(self.slider)

        self._snapshots = []

    # ── API pública ──────────────────────────────────────────

    def set_snapshots(self, snapshots):
        """
        Carga la lista de snapshots desde GeneticAlgorithm.gen_snapshots.
        """
        self._snapshots = snapshots
        self._total_gens = len(snapshots)
        if self._total_gens > 0:
            self.slider.min = 0
            self.slider.max = max(self._total_gens - 1, 1)
            self.slider.value = self._total_gens - 1
            self._update_info(self._total_gens - 1)

    def add_snapshot(self, snapshot):
        """Agrega un snapshot nuevo (durante entrenamiento en vivo)."""
        self._snapshots.append(snapshot)
        self._total_gens = len(self._snapshots)
        self.slider.max = max(self._total_gens - 1, 1)
        if not self._autoplay:
            self.slider.value = self._total_gens - 1
            self._update_info(self._total_gens - 1)

    def reset(self):
        """Limpia los snapshots."""
        self._snapshots = []
        self._total_gens = 0
        self.slider.value = 0
        self.slider.max = 1
        self.lbl_info.text = "GEN -- / --   SCORE --   FITNESS --"
        self._stop_autoplay()

    # ── Callbacks internos ───────────────────────────────────

    def _on_slider(self, instance, value):
        idx = int(value)
        self._update_info(idx)
        if self._on_gen_select and 0 <= idx < len(self._snapshots):
            self._on_gen_select(idx)

    def _update_info(self, idx):
        if not self._snapshots or idx >= len(self._snapshots):
            return
        s = self._snapshots[idx]
        self.lbl_info.text = (
            f"GEN {s['gen']} / {self._total_gens}   "
            f"SCORE {s['score']}   "
            f"FITNESS {s['fitness']:,.0f}"
        )

    def _jump(self, idx):
        idx = max(0, min(idx, self._total_gens - 1))
        self.slider.value = idx

    # ── Autoplay ─────────────────────────────────────────────

    def _toggle_autoplay(self):
        if self._autoplay:
            self._stop_autoplay()
        else:
            self._start_autoplay()

    def _start_autoplay(self):
        self._autoplay = True
        self.btn_play.text = "⏸  Pausa"
        self.btn_play.background_color = (0.30, 0.10, 0.10, 1)
        if self.slider.value >= self._total_gens - 1:
            self.slider.value = 0
        self._autoplay_event = Clock.schedule_interval(
            self._autoplay_step, self._autoplay_speed
        )

    def _stop_autoplay(self):
        self._autoplay = False
        self.btn_play.text = "▶  Auto"
        self.btn_play.background_color = (0.1, 0.2, 0.35, 1)
        if self._autoplay_event:
            self._autoplay_event.cancel()
            self._autoplay_event = None

    def _autoplay_step(self, dt):
        next_val = int(self.slider.value) + 1
        if next_val >= self._total_gens:
            self._stop_autoplay()
            return
        self.slider.value = next_val