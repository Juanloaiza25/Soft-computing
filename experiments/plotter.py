"""
EvoSnake — Visualización de experimentos
==========================================
Genera gráficas de evolución del fitness y comparación entre
distintas configuraciones del Algoritmo Genético.

Uso
---
# Desde la UI o desde consola:
>>> from experiments.plotter import Plotter
>>> p = Plotter()
>>> p.add_run("Baseline",        history_baseline)
>>> p.add_run("Alta mutación",   history_high_mut)
>>> p.show_fitness_curve()
>>> p.show_comparison()

# Ejecución directa (carga JSONs guardados por genetic_algorithm.py):
    python -m experiments.plotter
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# Paleta coherente con los colores del juego
PALETTE = [
    "#20FF85",   # verde neón  — baseline
    "#FF4D6D",   # rojo        — alta mutación
    "#4DAAFF",   # azul        — población grande
    "#FFD94D",   # amarillo    — más generaciones
    "#CC88FF",   # violeta     — extra
]

DARK_BG   = "#0F0F16"
GRID_COLOR = "#1E1E28"


class Plotter:
    """
    Acumula historiales de ejecuciones del GA y genera gráficas comparativas.

    Parámetros
    ----------
    dark_mode : bool — usar fondo oscuro (coherente con la app)
    """

    def __init__(self, dark_mode: bool = True):
        self.runs: list[dict] = []   # [{"label": str, "history": list[dict]}]
        self.dark_mode = dark_mode
        if dark_mode:
            plt.style.use("dark_background")

    # ── API pública ─────────────────────────────────────────────────────────

    def add_run(self, label: str, history: list) -> None:
        self.runs.append({"label": label, "history": history})

    def add_run_from_file(self, label: str, filepath: str) -> None:
        """Carga un historial desde un archivo JSON.
        Normaliza claves antiguas (best/avg/worst) al formato actual.
        """
        with open(filepath) as f:
            history = json.load(f)

        # Normalizar claves del formato antiguo al actual
        normalized = []
        for d in history:
            entry = dict(d)
            if "best_fitness" not in entry and "best" in entry:
                entry["best_fitness"] = entry.pop("best")
            if "avg_fitness" not in entry and "avg" in entry:
                entry["avg_fitness"] = entry.pop("avg")
            # Campos opcionales con fallback a 0
            entry.setdefault("avg_fitness",        0)
            entry.setdefault("best_score",         0)
            entry.setdefault("global_best_score",  0)
            entry.setdefault("avg_score",          0)
            entry.setdefault("max_lifetime",       0)
            entry.setdefault("avg_lifetime",       0)
            normalized.append(entry)

        self.add_run(label, normalized)

    def show_fitness_curve(self, save_path: str = None) -> None:
        """
        Grafica fitness máximo y promedio por generación
        para todas las ejecuciones registradas.
        """
        if not self.runs:
            print("No hay datos. Usa add_run() primero.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        self._style_axes(ax)

        for i, run in enumerate(self.runs):
            color  = PALETTE[i % len(PALETTE)]
            gens   = [d["gen"]  for d in run["history"]]
            bests  = [d["best_fitness"] for d in run["history"]]
            avgs   = [d["avg_fitness"]  for d in run["history"]]

            ax.plot(gens, bests, color=color, linewidth=2,
                    label=f"{run['label']} — máximo")
            ax.plot(gens, avgs,  color=color, linewidth=1,
                    linestyle="--", alpha=0.6,
                    label=f"{run['label']} — promedio")

        ax.set_title("Evolución del Fitness por Generación",
                     fontsize=14, pad=12)
        ax.set_xlabel("Generación")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Guardada en {save_path}")
        plt.show()

    def show_comparison(self, save_path: str = None) -> None:
        """
        Panel 2×2 comparando todas las ejecuciones:
          · Fitness máximo
          · Fitness promedio
          · Score máximo
          · Convergencia (generación donde best > 90% del máximo final)
        """
        if not self.runs:
            print("No hay datos.")
            return

        fig = plt.figure(figsize=(13, 8))
        fig.patch.set_facecolor(DARK_BG if self.dark_mode else "white")
        gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

        axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
        titles = [
            "Fitness máximo por generación",
            "Fitness promedio por generación",
            "Score máximo por generación",
            "Convergencia (% del máximo final)",
        ]

        for ax, title in zip(axes, titles):
            self._style_axes(ax)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Generación", fontsize=9)

        for i, run in enumerate(self.runs):
            color   = PALETTE[i % len(PALETTE)]
            h       = run["history"]
            label   = run["label"]
            gens    = [d["gen"]           for d in h]
            bests   = [d["best_fitness"]  for d in h]
            avgs    = [d["avg_fitness"]   for d in h]
            scores  = [d["best_score"]    for d in h]

            # Convergencia normalizada
            max_fit = max(bests) if max(bests) > 0 else 1
            conv    = [b / max_fit * 100 for b in bests]

            axes[0].plot(gens, bests,  color=color, linewidth=2, label=label)
            axes[1].plot(gens, avgs,   color=color, linewidth=2, label=label)
            axes[2].plot(gens, scores, color=color, linewidth=2, label=label)
            axes[3].plot(gens, conv,   color=color, linewidth=2, label=label)

        for ax in axes:
            ax.legend(fontsize=8)

        fig.suptitle("Comparación de Experimentos — EvoSnake GA",
                     fontsize=15, y=1.01,
                     color="white" if self.dark_mode else "black")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Guardada en {save_path}")
        plt.show()

    def show_final_bar(self, save_path: str = None) -> None:
        """
        Barras comparando el fitness máximo final de cada experimento.
        Útil para el informe / presentación.
        """
        if not self.runs:
            return

        labels = [r["label"]           for r in self.runs]
        finals = [max(d["best_fitness"] for d in r["history"]) for r in self.runs]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(self.runs))]

        fig, ax = plt.subplots(figsize=(8, 4))
        self._style_axes(ax)

        bars = ax.bar(labels, finals, color=colors, edgecolor="none", width=0.5)

        for bar, val in zip(bars, finals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(finals) * 0.01,
                f"{val:.0f}",
                ha="center", va="bottom", fontsize=10,
                color="white" if self.dark_mode else "black",
            )

        ax.set_title("Fitness Máximo Final por Configuración", fontsize=13)
        ax.set_ylabel("Fitness")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ── Nuevas gráficas ─────────────────────────────────────────────────────

    def show_score_evolution(self, save_path: str = None) -> None:
        """
        Evolución del score (comidas) a lo largo de las generaciones.
        Muestra: mejor score de la gen, score promedio y highscore acumulado.
        """
        if not self.runs:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        self._style_axes(ax)

        for i, run in enumerate(self.runs):
            color = PALETTE[i % len(PALETTE)]
            h     = run["history"]
            gens  = [d["gen"] for d in h]

            best_scores   = [d.get("best_score", 0)          for d in h]
            avg_scores    = [d.get("avg_score", 0)            for d in h]
            global_bests  = [d.get("global_best_score", 0)   for d in h]

            ax.plot(gens, global_bests, color=color, linewidth=2.5,
                    label=f"{run['label']} — highscore acumulado")
            ax.plot(gens, best_scores, color=color, linewidth=1.5,
                    linestyle="--",
                    label=f"{run['label']} — mejor de la gen")
            ax.plot(gens, avg_scores, color=color, linewidth=1,
                    linestyle=":", alpha=0.5,
                    label=f"{run['label']} — promedio de la gen")

        ax.set_title("Evolución del Score (Comidas) por Generación",
                     fontsize=14, pad=12)
        ax.set_xlabel("Generación")
        ax.set_ylabel("Score (manzanas comidas + 3 iniciales)")
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Guardada en {save_path}")
        plt.show()

    def show_lifetime_evolution(self, save_path: str = None) -> None:
        """
        Evolución del tiempo de vida (movimientos) de la serpiente.
        Muestra lifetime máximo y promedio por generación.
        """
        if not self.runs:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        self._style_axes(ax)

        for i, run in enumerate(self.runs):
            color = PALETTE[i % len(PALETTE)]
            h     = run["history"]
            gens  = [d["gen"] for d in h]

            max_lt = [d.get("max_lifetime", 0) for d in h]
            avg_lt = [d.get("avg_lifetime", 0) for d in h]

            ax.plot(gens, max_lt, color=color, linewidth=2,
                    label=f"{run['label']} — max lifetime")
            ax.plot(gens, avg_lt, color=color, linewidth=1,
                    linestyle="--", alpha=0.6,
                    label=f"{run['label']} — avg lifetime")

        ax.set_title("Evolución del Lifetime (Movimientos) por Generación",
                     fontsize=14, pad=12)
        ax.set_xlabel("Generación")
        ax.set_ylabel("Movimientos")
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Guardada en {save_path}")
        plt.show()

    def show_convergence_speed(self, threshold: float = 0.9,
                               save_path: str = None) -> None:
        """
        Muestra qué tan rápido converge cada ejecución.
        Marca la generación donde el fitness supera `threshold`×100% del máximo.
        También grafica la curva de convergencia normalizada con área sombreada.
        """
        if not self.runs:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        self._style_axes(ax)

        for i, run in enumerate(self.runs):
            color = PALETTE[i % len(PALETTE)]
            h     = run["history"]
            gens  = [d["gen"]          for d in h]
            bests = [d["best_fitness"] for d in h]

            max_fit = max(bests) if max(bests) > 0 else 1
            norm    = [b / max_fit for b in bests]

            ax.plot(gens, norm, color=color, linewidth=2,
                    label=run["label"])
            ax.fill_between(gens, norm, alpha=0.08, color=color)

            # Marcar generación de convergencia
            conv_gen = next(
                (g for g, n in zip(gens, norm) if n >= threshold),
                None
            )
            if conv_gen is not None:
                conv_val = norm[gens.index(conv_gen)]
                ax.axvline(conv_gen, color=color, linewidth=1,
                           linestyle=":", alpha=0.7)
                ax.annotate(
                    f"Gen {conv_gen}",
                    xy=(conv_gen, conv_val),
                    xytext=(conv_gen + max(gens) * 0.02, conv_val - 0.07),
                    color=color, fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                )

        ax.axhline(threshold, color="white", linewidth=0.8,
                   linestyle="--", alpha=0.4,
                   label=f"Umbral {int(threshold*100)}%")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Velocidad de Convergencia (umbral={int(threshold*100)}% del máximo)",
                     fontsize=14, pad=12)
        ax.set_xlabel("Generación")
        ax.set_ylabel("Fitness normalizado (0–1)")
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Guardada en {save_path}")
        plt.show()

    def show_fitness_gap(self, save_path: str = None) -> None:
        """
        Brecha entre fitness máximo y promedio por generación.
        Una brecha grande indica poca diversidad (dominancia de un individuo).
        Una brecha pequeña indica convergencia o exploración uniforme.
        """
        if not self.runs:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        self._style_axes(ax)

        for i, run in enumerate(self.runs):
            color = PALETTE[i % len(PALETTE)]
            h     = run["history"]
            gens  = [d["gen"]          for d in h]
            bests = [d["best_fitness"] for d in h]
            avgs  = [d["avg_fitness"]  for d in h]
            gaps  = [b - a for b, a in zip(bests, avgs)]

            ax.plot(gens, gaps, color=color, linewidth=2,
                    label=run["label"])
            ax.fill_between(gens, gaps, alpha=0.08, color=color)

        ax.set_title("Brecha Fitness Máximo − Promedio (Indicador de Diversidad)",
                     fontsize=13, pad=12)
        ax.set_xlabel("Generación")
        ax.set_ylabel("Fitness gap")
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Guardada en {save_path}")
        plt.show()

    def show_score_distribution(self, bins: int = 15,
                                save_path: str = None) -> None:
        """
        Histograma del score promedio por generación.
        Muestra cómo se distribuye el rendimiento a lo largo del entrenamiento.
        """
        if not self.runs:
            return

        n     = len(self.runs)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
        if n == 1:
            axes = [axes]

        fig.patch.set_facecolor(DARK_BG if self.dark_mode else "white")

        for i, (run, ax) in enumerate(zip(self.runs, axes)):
            color  = PALETTE[i % len(PALETTE)]
            scores = [d.get("best_score", 0) for d in run["history"]]

            self._style_axes(ax)
            ax.hist(scores, bins=bins, color=color, edgecolor="none", alpha=0.85)
            ax.axvline(np.mean(scores), color="white", linewidth=1.2,
                       linestyle="--", label=f"Media: {np.mean(scores):.1f}")
            ax.set_title(f"{run['label']}\nDistribución de Scores",
                         fontsize=11)
            ax.set_xlabel("Score")
            ax.set_ylabel("Frecuencia (generaciones)")
            ax.legend(fontsize=8)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Guardada en {save_path}")
        plt.show()

    def show_dashboard(self, save_path: str = None) -> None:
        """
        Dashboard completo 3×2 con todas las métricas clave en una sola figura.
        Ideal para el informe / presentación del proyecto.
        """
        if not self.runs:
            print("No hay datos.")
            return

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor(DARK_BG if self.dark_mode else "white")
        gs = gridspec.GridSpec(3, 2, hspace=0.50, wspace=0.30)

        ax_fitness   = fig.add_subplot(gs[0, 0])
        ax_score     = fig.add_subplot(gs[0, 1])
        ax_lifetime  = fig.add_subplot(gs[1, 0])
        ax_conv      = fig.add_subplot(gs[1, 1])
        ax_gap       = fig.add_subplot(gs[2, 0])
        ax_avg_score = fig.add_subplot(gs[2, 1])

        panels = [
            (ax_fitness,   "Fitness Máximo y Promedio"),
            (ax_score,     "Score Máximo (Highscore)"),
            (ax_lifetime,  "Lifetime Máximo (Movimientos)"),
            (ax_conv,      "Convergencia Normalizada"),
            (ax_gap,       "Brecha Fitness Max−Avg (Diversidad)"),
            (ax_avg_score, "Score Promedio de la Población"),
        ]
        for ax, title in panels:
            self._style_axes(ax)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Generación", fontsize=8)

        for i, run in enumerate(self.runs):
            color = PALETTE[i % len(PALETTE)]
            h     = run["history"]
            label = run["label"]
            gens  = [d["gen"]                    for d in h]
            bests = [d["best_fitness"]            for d in h]
            avgs  = [d["avg_fitness"]             for d in h]
            bsc   = [d.get("best_score", 0)       for d in h]
            gbs   = [d.get("global_best_score", 0)for d in h]
            mlt   = [d.get("max_lifetime", 0)     for d in h]
            ascr  = [d.get("avg_score", 0)        for d in h]

            max_fit = max(bests) if max(bests) > 0 else 1
            norm    = [b / max_fit for b in bests]
            gaps    = [b - a for b, a in zip(bests, avgs)]

            # Panel 1: Fitness
            ax_fitness.plot(gens, bests, color=color, linewidth=1.8,
                            label=f"{label} — max")
            ax_fitness.plot(gens, avgs, color=color, linewidth=1,
                            linestyle="--", alpha=0.5)

            # Panel 2: Score / Highscore
            ax_score.plot(gens, gbs, color=color, linewidth=2,
                          label=f"{label}")
            ax_score.plot(gens, bsc, color=color, linewidth=1,
                          linestyle="--", alpha=0.5)

            # Panel 3: Lifetime
            ax_lifetime.plot(gens, mlt, color=color, linewidth=1.8,
                             label=label)

            # Panel 4: Convergencia
            ax_conv.plot(gens, norm, color=color, linewidth=1.8,
                         label=label)
            ax_conv.fill_between(gens, norm, alpha=0.07, color=color)

            # Marcar gen de convergencia al 90%
            conv_gen = next(
                (g for g, n in zip(gens, norm) if n >= 0.9), None
            )
            if conv_gen is not None:
                ax_conv.axvline(conv_gen, color=color,
                                linewidth=0.8, linestyle=":", alpha=0.7)

            # Panel 5: Gap
            ax_gap.plot(gens, gaps, color=color, linewidth=1.8,
                        label=label)
            ax_gap.fill_between(gens, gaps, alpha=0.07, color=color)

            # Panel 6: Avg score
            ax_avg_score.plot(gens, ascr, color=color, linewidth=1.8,
                              label=label)

        ax_conv.axhline(0.9, color="white", linewidth=0.7,
                        linestyle="--", alpha=0.35, label="Umbral 90%")
        ax_conv.set_ylim(0, 1.05)

        for ax, _ in panels:
            ax.legend(fontsize=7)

        title_color = "white" if self.dark_mode else "black"
        fig.suptitle("Dashboard de Entrenamiento — EvoSnake GA",
                     fontsize=16, color=title_color, y=1.01)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Dashboard guardado en {save_path}")
        plt.show()

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _style_axes(self, ax) -> None:
        """Aplica estilo oscuro coherente con la app."""
        if self.dark_mode:
            ax.set_facecolor(GRID_COLOR)
            ax.tick_params(colors="0.7")
            ax.xaxis.label.set_color("0.7")
            ax.yaxis.label.set_color("0.7")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333344")
        ax.grid(True, color="#2A2A3A" if self.dark_mode else "#EEEEEE",
                linewidth=0.5)


# ── Ejecución directa ────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Carga history.json y genera todas las gráficas disponibles.
    Ejecutar desde la raíz del proyecto:
        python -m experiments.plotter
    """
    plotter = Plotter(dark_mode=True)
    experiments_dir = os.path.join(os.path.dirname(__file__))

    # ── Intentar cargar history.json principal ────────────────────────
    main_history = os.path.join(experiments_dir, "history.json")
    if os.path.exists(main_history):
        plotter.add_run_from_file("Entrenamiento", main_history)
        print(f"✓ Cargado: {main_history}")
    else:
        print(f"No se encontró {main_history}")

    # ── Intentar cargar historiales de experimentos adicionales ───────
    json_files = {
        "Baseline (pop=50, mut=0.1)":         "history_baseline.json",
        "Alta mutación (pop=50, mut=0.3)":    "history_high_mut.json",
        "Población grande (pop=100, mut=0.1)": "history_large_pop.json",
        "Más generaciones (pop=50, gen=100)":  "history_more_gens.json",
    }
    for label, filename in json_files.items():
        path = os.path.join(experiments_dir, filename)
        if os.path.exists(path):
            plotter.add_run_from_file(label, path)
            print(f"✓ Cargado: {label}")

    if not plotter.runs:
        print("No se encontraron archivos JSON.")
        print("Primero entrena con:  python -m ai.genetic_algorithm")
    else:
        out = experiments_dir
        print(f"\nGenerando gráficas para {len(plotter.runs)} run(s)...\n")

        plotter.show_fitness_curve(
            save_path=os.path.join(out, "fitness_curve.png"))
        plotter.show_score_evolution(
            save_path=os.path.join(out, "score_evolution.png"))
        plotter.show_lifetime_evolution(
            save_path=os.path.join(out, "lifetime_evolution.png"))
        plotter.show_convergence_speed(
            save_path=os.path.join(out, "convergence.png"))
        plotter.show_fitness_gap(
            save_path=os.path.join(out, "fitness_gap.png"))
        plotter.show_score_distribution(
            save_path=os.path.join(out, "score_distribution.png"))
        plotter.show_dashboard(
            save_path=os.path.join(out, "dashboard.png"))

        if len(plotter.runs) > 1:
            plotter.show_comparison(
                save_path=os.path.join(out, "comparison.png"))
            plotter.show_final_bar(
                save_path=os.path.join(out, "final_bar.png"))