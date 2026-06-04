"""
EvoSnake — Punto de entrada
============================
Ejecutar:
    python main.py
"""

import os
import json
import numpy as np

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout

from ui.menu_screen import MenuScreen
from ui.snake_screen import SnakeScreen
from ai.neural_network import NeuralNetwork
from utils.constants import GRID_SIZE, CELL_SIZE


# Rutas de modelos
BEST_WEIGHTS_PATH = os.path.join("models", "best_weights.npy")
HISTORY_PATH      = os.path.join("experiments", "history.json")
SNAPSHOTS_PATH     = os.path.join("models", "generation_snapshots.json")

# Tamaño de ventana simulando dispositivo móvil
GAME_WIN_W = 380
GAME_WIN_H = 740


def load_best_nn():
    """
    Carga la mejor red neuronal desde best_weights.npy.
    Retorna (NeuralNetwork, model_info, history) o (None, None, None).
    """
    if not os.path.exists(BEST_WEIGHTS_PATH):
        return None, None, None

    weights = np.load(BEST_WEIGHTS_PATH)
    nn = NeuralNetwork()

    if weights.size != nn.chromosome_size():
        print(
            f"Tamano de pesos ({weights.size}) no coincide "
            f"con la arquitectura actual ({nn.chromosome_size()})."
        )
        return None, None, None

    nn.set_flat_weights(weights)

    # Cargar historial completo para playback gen 1 -> N
    history = []
    model_info = {"gen": "--", "best_fitness": 0, "highscore": 0}
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)
            if history:
                model_info["gen"] = history[-1].get("gen", "--")
                model_info["best_fitness"] = max(
                    h.get("best_fitness", 0) for h in history
                )
                model_info["highscore"] = max(
                    h.get("global_best_score", 0) for h in history
                )
        except Exception:
            pass

    print(f"Modelo cargado desde {BEST_WEIGHTS_PATH} "
          f"({len(history)} generaciones)")
    return nn, model_info, history


class EvoSnakeApp(App):

    def build(self):
        Window.size = (GAME_WIN_W, GAME_WIN_H)
        Window.title = "EvoSnake"
        Window.clearcolor = (0.12, 0.08, 0.22, 1)

        self._container = FloatLayout()
        self._current_screen = None
        self._show_menu()
        return self._container

    # ── Navegación ──────────────────────────────────

    def _show_menu(self):
        """Muestra el menú principal."""
        Window.size = (GAME_WIN_W, GAME_WIN_H)
        Window.clearcolor = (0.12, 0.08, 0.22, 1)

        menu = MenuScreen(
            on_manual=self._start_manual,
            on_ai=self._start_ai,
            on_graphs=self._show_graphs,
        )
        self._switch_screen(menu)

    def _start_manual(self):
        """Inicia el juego en modo manual."""
        Window.size = (GAME_WIN_W, GAME_WIN_H)
        Window.clearcolor = (0.04, 0.04, 0.07, 1)

        screen = SnakeScreen(on_back=self._show_menu)
        self._switch_screen(screen)

    def _start_ai(self):
            """Inicia el juego en modo IA con modelo cargado."""
            Window.size = (GAME_WIN_W, GAME_WIN_H)
            Window.clearcolor = (0.04, 0.04, 0.07, 1)

            best_nn, model_info, history = load_best_nn()
            screen = SnakeScreen(on_back=self._show_menu)

            # Cargar snapshots PRIMERO
            if os.path.exists(SNAPSHOTS_PATH):
                try:
                    with open(SNAPSHOTS_PATH) as f:
                        raw = json.load(f)
                    snapshots = []
                    for s in raw:
                        s["weights"] = np.array(s["weights"], dtype=np.float64)
                        snapshots.append(s)
                    screen.gen_bar._snapshots = snapshots
                    screen.gen_bar._total_gens = len(snapshots)
                    print(f"Snapshots cargados: {len(snapshots)}")
                except Exception as e:
                    print(f"Error cargando snapshots: {e}")
            else:
                print(f"No existe archivo: {SNAPSHOTS_PATH}")

            # Luego cargar modelo
            if best_nn is not None:
                screen.load_ai_model(best_nn, model_info)
            else:
                screen._toggle_ai()

            self._switch_screen(screen)

    def _show_graphs(self):
        """Abre gráficas matplotlib en hilos separados."""
        if not os.path.exists(HISTORY_PATH):
            print("No hay datos de historial")
            return

        from experiments.plotter import Plotter
        plotter = Plotter(dark_mode=True)
        plotter.add_run_from_file("Entrenamiento", HISTORY_PATH)
        plotter.show_fitness_curve()
        plotter.show_score_evolution()
        plotter.show_lifetime_evolution()
        plotter.show_convergence_speed()
        plotter.show_fitness_gap()
        plotter.show_score_distribution()
        plotter.show_dashboard()

    def _switch_screen(self, new_widget):
        """Reemplaza la pantalla actual."""
        if self._current_screen is not None:
            if hasattr(self._current_screen, '_clock_event'):
                self._current_screen._clock_event.cancel()
            if (hasattr(self._current_screen, '_ga')
                    and self._current_screen._ga):
                self._current_screen._ga.request_stop()
            self._container.remove_widget(self._current_screen)

        self._current_screen = new_widget
        new_widget.size_hint = (1, 1)
        self._container.add_widget(new_widget)


if __name__ == "__main__":
    EvoSnakeApp().run()