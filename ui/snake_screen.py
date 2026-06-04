"""
EvoSnake — Pantalla principal (Kivy)
=====================================
Layout moderno: tablero a la izquierda, visualización de red neuronal
a la derecha con estadísticas en tiempo real.

Controles:
  · Flechas / WASD  → mover la serpiente (modo manual)
  · Botones         → reiniciar, modo IA, gráficas, menú
"""

import threading
import numpy as np

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.metrics import dp

from game.snake_game import SnakeGame
from ui.snake_board import SnakeBoard
from ui.nn_widget import NNVisualization
from ai.neural_network import NeuralNetwork
from ai.genetic_algorithm import GeneticAlgorithm
from utils.constants import (
    GRID_SIZE, CELL_SIZE, FPS_PLAY, FPS_AI, COLORS,
    POP_SIZE, MUTATION_RATE
)
from ui.gen_replay_bar import GenReplayBar


class StyledButton(Button):
    """Botón con esquinas redondeadas."""

    def __init__(self, bg_color=(0.12, 0.12, 0.18, 1), **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)
        self.background_normal = ''
        self.background_down = ''
        self._bg_color = bg_color
        self.bold = True
        self.font_size = dp(12)
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self._bg_color)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(8)])


class SnakeScreen(BoxLayout):
    """
    Pantalla principal con layout moderno:
      · Izquierda: tablero del juego
      · Derecha: panel de NN + estadísticas
      · Inferior: barra de botones
    """

    def __init__(self, on_back=None, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.padding = dp(10)
        self._on_back = on_back
        self.spacing = dp(8)

        self.game = SnakeGame(grid_size=GRID_SIZE)

        # Estado del modo IA
        self._mode = "manual"
        self._best_nn = None
        self._ga = None
        self._ga_history = []
        self._current_gen = 0
        self._best_fitness = 0
        self._highscore = 0
        self._replay_state = None
        self._waiting_restart = False

        # ═══════════════════════════════════════════════
        #  CONTENIDO PRINCIPAL (vertical para mobile)
        # ═══════════════════════════════════════════════
        main_content = BoxLayout(
            orientation="vertical",
            spacing=dp(12),
        )

        # ── Panel superior: tablero + score ──────────
        left_panel = BoxLayout(
            orientation="vertical",
            size_hint=(1, 1.5),  # Ocupa más espacio verticalmente
            spacing=dp(6),
        )

        # Score label encima del tablero
        self.lbl_score = Label(
            text="SCORE : 0",
            font_size=dp(22),
            bold=True,
            color=(1, 1, 1, 1),
            size_hint=(1, None),
            height=dp(34),
            halign="center",
            valign="middle",
        )
        self.lbl_score.bind(size=self.lbl_score.setter("text_size"))

        # Tablero del juego
        board_anchor = AnchorLayout(
            anchor_x="center",
            anchor_y="center",
            size_hint=(1, 1),
        )
        self.board = SnakeBoard(self.game)
        board_anchor.add_widget(self.board)

        # Highscore debajo del tablero
        self.lbl_highscore = Label(
            text="HIGHSCORE : 0",
            font_size=dp(14),
            bold=True,
            color=(0.5, 0.5, 0.6, 1),
            size_hint=(1, None),
            height=dp(24),
            halign="center",
            valign="middle",
        )
        self.lbl_highscore.bind(
            size=self.lbl_highscore.setter("text_size")
        )

        left_panel.add_widget(self.lbl_score)
        left_panel.add_widget(board_anchor)
        left_panel.add_widget(self.lbl_highscore)

        # ── Panel inferior: stats + NN + leyenda ───────
        right_panel = BoxLayout(
            orientation="vertical",
            spacing=dp(4),
            size_hint=(1, 1),
        )

        # Stats en la parte superior del panel derecho
        stats_box = BoxLayout(
            orientation="vertical",
            size_hint=(1, None),
            height=dp(80),
            padding=[dp(8), dp(4)],
        )

        self.lbl_gen = Label(
            text="GEN : --",
            font_size=dp(13),
            bold=True,
            color=(0.7, 0.7, 0.8, 1),
            size_hint=(1, None),
            height=dp(18),
            halign="left",
            valign="middle",
        )
        self.lbl_gen.bind(size=self.lbl_gen.setter("text_size"))

        self.lbl_fitness = Label(
            text="BEST FITNESS : --",
            font_size=dp(13),
            bold=True,
            color=(0.7, 0.7, 0.8, 1),
            size_hint=(1, None),
            height=dp(18),
            halign="left",
            valign="middle",
        )
        self.lbl_fitness.bind(size=self.lbl_fitness.setter("text_size"))

        self.lbl_moves = Label(
            text="MOVES LEFT : 200",
            font_size=dp(13),
            bold=True,
            color=(0.7, 0.7, 0.8, 1),
            size_hint=(1, None),
            height=dp(18),
            halign="left",
            valign="middle",
        )
        self.lbl_moves.bind(size=self.lbl_moves.setter("text_size"))

        self.lbl_mutation = Label(
            text=f"MUTATION RATE : {MUTATION_RATE}",
            font_size=dp(13),
            bold=True,
            color=(0.7, 0.7, 0.8, 1),
            size_hint=(1, None),
            height=dp(18),
            halign="left",
            valign="middle",
        )
        self.lbl_mutation.bind(
            size=self.lbl_mutation.setter("text_size")
        )

        stats_box.add_widget(self.lbl_gen)
        stats_box.add_widget(self.lbl_fitness)
        stats_box.add_widget(self.lbl_moves)
        stats_box.add_widget(self.lbl_mutation)

        # NN Visualization (ocupa el espacio restante)
        self.nn_widget = NNVisualization(size_hint=(1, 1))

        right_panel.add_widget(stats_box)
        # El NN Widget puede verse apretado en mobile, pero permitimos que tome el espacio sobrante
        right_panel.add_widget(self.nn_widget)

         # ── Barra de replay por generación ────────────────────────
        self.gen_bar = GenReplayBar(
            on_gen_select=self._on_gen_selected,
        )
        self.gen_bar.opacity = 0
        self.gen_bar.disabled = True
        right_panel.add_widget(self.gen_bar)
        
        main_content.add_widget(left_panel)
        main_content.add_widget(right_panel)

        # ═══════════════════════════════════════════════
        #  BARRA INFERIOR (botones) - AHORA ARRIBA
        # ═══════════════════════════════════════════════
        btn_bar = BoxLayout(
            size_hint=(1, None),
            height=dp(42),
            spacing=dp(8),
            padding=[0, dp(4)],
        )

        btn_back = StyledButton(
            text="<  Menu",
            bg_color=(0.18, 0.12, 0.28, 1),
            color=(0.8, 0.7, 0.9, 1),
            size_hint=(0.22, 1),
        )
        btn_back.bind(on_press=lambda _: self._go_back())

        btn_reset = StyledButton(
            text="Reiniciar",
            bg_color=(0.10, 0.10, 0.16, 1),
            color=(0.8, 0.8, 0.85, 1),
            size_hint=(0.22, 1),
        )
        btn_reset.bind(on_press=lambda _: self.reset_game())

        self.btn_ai = StyledButton(
            text="Modo IA",
            bg_color=(0.08, 0.22, 0.15, 1),
            color=(0.20, 1.00, 0.65, 1),
            size_hint=(0.30, 1),
        )
        self.btn_ai.bind(on_press=lambda _: self._toggle_ai())

        self.btn_fitness = StyledButton(
            text="Graficos",
            bg_color=(0.10, 0.10, 0.16, 1),
            color=(0.8, 0.8, 0.85, 1),
            size_hint=(0.26, 1),
        )
        self.btn_fitness.bind(on_press=lambda _: self._show_fitness())

        btn_bar.add_widget(btn_back)
        btn_bar.add_widget(btn_reset)
        btn_bar.add_widget(self.btn_ai)
        btn_bar.add_widget(self.btn_fitness)

        # ── Añadir al layout principal (BOTONES ARRIBA) ──
        self.add_widget(btn_bar)
        self.add_widget(main_content)

        # ── Teclado ─────────────────────────────
        Window.bind(on_key_down=self._on_key)

        # ── Loop del juego ──────────────────────
        self._clock_event = Clock.schedule_interval(
            self._update, 1 / FPS_PLAY
        )

        # Dibujo inicial
        self.board.draw()
        self.nn_widget.draw()

    # ── Carga de modelo IA ───────────────────────────

    def load_ai_model(self, nn, model_info=None, history=None):
            """Carga snapshots y empieza replay desde gen 1."""
            self._best_nn = nn
            self._mode = "replay"
            self._snapshot_index = 0

            # Cancelar timer anterior si existe
            if getattr(self, '_history_timer', None):
                self._history_timer.cancel()
                self._history_timer = None

            if model_info:
                self._highscore = model_info.get("highscore", 0)
                self.lbl_highscore.text = f"HIGHSCORE : {self._highscore}"

            self.btn_ai.text = "Manual"
            self.btn_ai._bg_color = (0.15, 0.15, 0.28, 1)

            # Mostrar barra de replay
            if self.gen_bar._snapshots:
                self.gen_bar.opacity = 1
                self.gen_bar.disabled = False
                self.gen_bar.set_snapshots(self.gen_bar._snapshots)

            self._clock_event.cancel()
            self._clock_event = Clock.schedule_interval(self._update, 1 / FPS_AI)

            # Cargar primera generación
            self._load_snapshot(0)

    def _load_snapshot(self, idx):
        """Carga la NN del snapshot idx y reinicia el juego."""
        snapshots = self.gen_bar._snapshots
        if not snapshots or idx >= len(snapshots):
            return

        self._snapshot_index = idx
        snap = snapshots[idx]

        nn = NeuralNetwork()
        nn.set_flat_weights(snap["weights"])
        self._best_nn = nn
        self.nn_widget.set_nn(nn)

        self.lbl_gen.text = f"GEN : {snap['gen']}"
        self.lbl_fitness.text = f"BEST FITNESS : {snap['fitness']:,.0f}"
        self.lbl_score.text = f"SCORE : {snap['score']}"

        # Sincronizar el slider visualmente sin gatillar el callback de nuevo
        self.gen_bar.slider.value = idx

        # Configurar la comida exacta que tuvo en el entrenamiento
        self.game.replay_food_list = snap.get("food_list", [])
        self.game.reset()
        self.board.draw()

    def _advance_history(self, dt):
        """Avanza una generación en el historial de playback."""
        if self._history_index >= len(self._history_playback):
            # Llegamos al final, detener timer
            if self._history_timer:
                self._history_timer.cancel()
                self._history_timer = None
            return

        entry = self._history_playback[self._history_index]
        self._history_index += 1

        gen = entry.get("gen", self._history_index)
        best_fit = entry.get("best_fitness", 0)
        best_score = entry.get("global_best_score", 0)

        self.lbl_gen.text = f"GEN : {gen}"
        self.lbl_fitness.text = f"BEST FITNESS : {best_fit:,.0f}"
        self._highscore = max(self._highscore, best_score)
        self.lbl_highscore.text = f"HIGHSCORE : {self._highscore}"
        self._current_gen = gen

    def _go_back(self):
        """Regresa al menú principal."""
        if self._mode == "training" and self._ga:
            self._ga.request_stop()
        self._clock_event.cancel()
        if getattr(self, '_history_timer', None):
            self._history_timer.cancel()
        if self._on_back:
            self._on_back()

    # ── Control ──────────────────────────────────────

    def reset_game(self):
        """Reinicia el juego y vuelve al modo manual."""
        if self._mode == "training" and self._ga:
            self._ga.request_stop()
        if getattr(self, '_history_timer', None):
            self._history_timer.cancel()
            self._history_timer = None

        self._mode = "manual"
        self._best_nn = None
        self.game.replay_food_list = None
        self.game.reset()
        self._refresh_labels()
        self.board.draw()
        self.nn_widget.draw()

        self._clock_event.cancel()
        self._clock_event = Clock.schedule_interval(
            self._update, 1 / FPS_PLAY
        )

        self.btn_ai.text = "Modo IA"
        self.btn_ai._bg_color = (0.08, 0.22, 0.15, 1)

    # ── Modo IA ──────────────────────────────────────

    def _toggle_ai(self):
        """Alterna entre iniciar/detener IA."""
        if self._mode == "manual":
            self._start_training()
        elif self._mode == "training":
            if self._ga:
                self._ga.request_stop()
        elif self._mode == "replay":
            self.reset_game()

    def _start_training(self):
            """Inicia el entrenamiento del GA en un hilo separado."""
            self._mode = "training"
            self._current_gen = 0
            self._ga_history = []
            self.gen_bar.reset()
            self.gen_bar.opacity = 0
            self.gen_bar.disabled = True

            self.lbl_gen.text = "GEN : 0"
            self.btn_ai.text = "Detener"
            self.btn_ai._bg_color = (0.30, 0.10, 0.10, 1)

            def on_generation(gen, stats, best_ind):
                self._current_gen = gen
                self._ga_history.append(stats)
                snapshot = None
                if self._ga and self._ga.gen_snapshots:
                    snapshot = self._ga.gen_snapshots[-1]
                Clock.schedule_once(
                    lambda dt: self._update_training_ui(gen, stats, best_ind, snapshot)
                )

            self._ga = GeneticAlgorithm(
                on_generation=on_generation,
                parallel=False,
            )

            def train():
                best_nn, history = self._ga.evolve(generations=500)
                self._best_nn = best_nn
                self._ga_history = history
                Clock.schedule_once(lambda dt: self._after_training())

            thread = threading.Thread(target=train, daemon=True)
            thread.start()

    def _update_training_ui(self, gen, stats, best_ind, snapshot=None):
            """Actualiza UI durante entrenamiento."""
            self.lbl_gen.text = f"GEN : {gen}"
            self.lbl_fitness.text = f"BEST FITNESS : {stats['best_fitness']:,.0f}"
            self.lbl_score.text = f"SCORE : {stats['best_score']}"
            self._highscore = max(self._highscore, stats.get('global_best_score', 0))
            self.lbl_highscore.text = f"HIGHSCORE : {self._highscore}"

            if snapshot is not None:
                self.gen_bar.opacity = 1
                self.gen_bar.disabled = False
                self.gen_bar.add_snapshot(snapshot)

            nn = NeuralNetwork()
            nn.set_flat_weights(np.array(best_ind, dtype=np.float64))
            self._current_gen_nn = nn
            self.nn_widget.set_nn(nn)
            self.game.reset()
            self.board.draw()

    def _start_replay(self):
        """Inicia reproducción de la mejor serpiente."""
        if self._best_nn is None:
            self.reset_game()
            return
        self._after_training()

    def _after_training(self):
        """Llamado cuando el GA termina. Activa replay de la última gen."""
        if self._best_nn is None:
            self.reset_game()
            return

        self._mode = "replay"
        self.game.reset()
        self.nn_widget.set_nn(self._best_nn)
        self.lbl_gen.text = f"GEN : {self._current_gen}"
        self.btn_ai.text = "Manual"
        self.btn_ai._bg_color = (0.15, 0.15, 0.28, 1)

        if self._ga and self._ga.gen_snapshots:
            self.gen_bar.set_snapshots(self._ga.gen_snapshots)
            self.gen_bar.opacity = 1
            self.gen_bar.disabled = False

        self._clock_event.cancel()
        self._clock_event = Clock.schedule_interval(self._update, 1 / FPS_AI)

    def _on_gen_selected(self, idx):
        """Carga el snapshot de la gen seleccionada y lanza el replay."""
        snapshots = None
        if self._ga and self._ga.gen_snapshots:
            snapshots = self._ga.gen_snapshots
        elif self.gen_bar._snapshots:
            snapshots = self.gen_bar._snapshots

        if not snapshots or idx >= len(snapshots):
            return

        snap = snapshots[idx]
        nn = NeuralNetwork()
        nn.set_flat_weights(snap["weights"])

        self.lbl_gen.text = f"GEN : {snap['gen']}"
        self.lbl_fitness.text = f"BEST FITNESS : {snap['fitness']:,.0f}"
        self.lbl_score.text = f"SCORE : {snap['score']}"

        self._best_nn = nn
        self._snapshot_index = idx          # ← para que _replay_restart avance bien
        self._mode = "replay"
        self.nn_widget.set_nn(nn)

        # ↓ LA LÍNEA QUE FALTABA
        self.game.replay_food_list = snap.get("food_list", [])
        self.game.reset()
        self.board.draw()

        self._clock_event.cancel()
        self._clock_event = Clock.schedule_interval(self._update, 1 / FPS_AI)

    # ── Fitness graph ────────────────────────────────

    def _show_fitness(self):
        """Abre gráfica de fitness (Matplotlib)."""
        if not self._ga_history:
            # Intentar cargar desde archivo
            import os
            import json
            path = os.path.join("experiments", "history.json")
            if os.path.exists(path):
                with open(path) as f:
                    self._ga_history = json.load(f)
            else:
                return

        from experiments.plotter import Plotter
        plotter = Plotter(dark_mode=True)
        plotter.add_run("Entrenamiento", self._ga_history)
        plotter.show_fitness_curve()

    # ── Loop ─────────────────────────────────────────

    def _update(self, dt):
            """Callback ejecutado FPS veces por segundo."""
            if self.game.dead:
                if self._mode == "replay" and not getattr(self, '_waiting_restart', False):
                    self._waiting_restart = True
                    Clock.schedule_once(
                        lambda dt: self._do_restart(), 1.5
                    )
                return

            active_nn = None

            if self._mode == "manual":
                self.game.move()
            elif self._mode == "replay" and self._best_nn:
                self.game.look()
                action = self._best_nn.predict(self.game.vision)
                self.game.set_direction(action)
                self.game.move()
                active_nn = self._best_nn
            elif (self._mode == "training"
                and getattr(self, "_current_gen_nn", None)):
                if self.game.dead:
                    self.game.reset()
                self.game.look()
                action = self._current_gen_nn.predict(self.game.vision)
                self.game.set_direction(action)
                self.game.move()
                active_nn = self._current_gen_nn

            self.board.draw()
            self._refresh_labels()

            if active_nn is not None:
                self.nn_widget.draw()

    def _do_restart(self):
        """Ejecuta el restart y resetea el flag."""
        self._waiting_restart = False
        self._replay_restart()

    # ── UI helpers ───────────────────────────────────

    def _refresh_labels(self):
        """Actualiza labels con el estado actual."""
        self.lbl_score.text = f"SCORE : {self.game.score}"
        self.lbl_moves.text = f"MOVES LEFT : {self.game.life_left}"

        if self.game.dead:
            fit = self.game.calculate_fitness()
            self.lbl_fitness.text = f"BEST FITNESS : {fit:,.0f}"

    def _replay_restart(self):
            """Al morir, avanza a la siguiente generación."""
            if self._mode != "replay":
                return

            next_idx = getattr(self, '_snapshot_index', 0) + 1
            snapshots = self.gen_bar._snapshots

            if snapshots and next_idx < len(snapshots):
                self._load_snapshot(next_idx)
            else:
                # Llegó al final, volver a gen 1
                self._load_snapshot(0)

    # ── Teclado ──────────────────────────────────────

    def _on_key(self, window, key, scancode, codepoint, modifier):
        """Mapea teclas a direcciones (modo manual)."""
        if self._mode != "manual":
            return

        keymap = {
            273: 1,   119: 1,    # UP / W
            274: 0,   115: 0,    # DOWN / S
            276: 2,    97: 2,    # LEFT / A
            275: 3,   100: 3,    # RIGHT / D
        }
        if key in keymap:
            self.game.set_absolute_direction(keymap[key])
