"""
SnakeAI — Algoritmo Genético con DEAP
========================================
Traducción fiel de Population.pde.

Implementa:
  - Población de 2000 serpientes
  - Fitness: lifetime² × 2^score (original)
  - Selección por ruleta (fitness-proportionate) como selectParent()
  - Elitismo: el mejor se preserva en la posición 0
  - Crossover de un punto + mutación gaussiana
  - Tracking del mejor snake para replay
  - Evaluación paralela con multiprocessing (opcional)
"""

import os
import json
import time
import random
import numpy as np
import multiprocessing as mp
import bisect
from copy import deepcopy

from deap import base, creator, tools

from game.snake_game import SnakeGame
from ai.neural_network import NeuralNetwork
from utils.constants import POP_SIZE, MUTATION_RATE, GRID_SIZE, GENERATIONS


# ── DEAP: crear tipos una sola vez ──────────────────────────────────────────
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


# ── Worker para multiprocessing (módulo-level, pickleable) ──────────────────
def _evaluate_snake_worker(flat_weights):
    """
    Evalúa una serpiente a partir de su cromosoma (vector plano de pesos).
    Función a nivel de módulo para que multiprocessing pueda serializarla.
    Retorna (fitness, score, lifetime, food_list).
    """
    nn = NeuralNetwork()
    nn.set_flat_weights(np.array(flat_weights, dtype=np.float64))
    game = SnakeGame(GRID_SIZE)

    while not game.dead:
        game.look()
        decision = nn.output(game.vision)
        action = int(np.argmax(decision))
        game.set_direction(action)
        game.move()

    fitness = game.calculate_fitness()
    return (fitness, game.score, game.lifetime, list(game.food_list))


class GeneticAlgorithm:
    """
    Algoritmo Genético fiel al Population.pde original.
    Usa DEAP para la infraestructura evolutiva.
    """

    def __init__(self, pop_size=POP_SIZE, mutation_rate=MUTATION_RATE,
                 seed=None, on_generation=None, parallel=False):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.on_generation = on_generation
        self.parallel = parallel
        self._stop_requested = False
        self._best_individual = None
        self._history = []
        self.gen_snapshots = []   # lista de dicts {gen, weights, score, fitness, food_list}
        

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Tamaño del cromosoma
        self._nn_template = NeuralNetwork()
        self._chrom_size = self._nn_template.chromosome_size()

        # DEAP toolbox
        self._toolbox = base.Toolbox()
        self._toolbox.register("attr_float", random.uniform, -1.0, 1.0)
        self._toolbox.register(
            "individual", tools.initRepeat, creator.Individual,
            self._toolbox.attr_float, n=self._chrom_size
        )
        self._toolbox.register(
            "population", tools.initRepeat, list, self._toolbox.individual
        )

    def request_stop(self):
        """Solicita detener la evolución."""
        self._stop_requested = True

    def _make_nn(self, individual):
        """Crea una NN a partir de un cromosoma DEAP."""
        nn = NeuralNetwork()
        nn.set_flat_weights(np.array(individual, dtype=np.float64))
        return nn

    def _evaluate_snake(self, individual):
        """
        Evalúa una serpiente: simula una partida completa (modo serial).
        Retorna (fitness,) como tupla para DEAP.
        """
        nn = self._make_nn(individual)
        game = SnakeGame(GRID_SIZE)

        while not game.dead:
            game.look()
            decision = nn.output(game.vision)
            action = int(np.argmax(decision))
            game.set_direction(action)
            game.move()

        fitness = game.calculate_fitness()

        # Guardar score en el individuo para tracking
        individual.game_score = game.score
        individual.game_lifetime = game.lifetime
        individual.food_list = list(game.food_list)

        return (fitness,)

    def _build_roulette(self, population):
            fitness_values = [
                max(1.0, ind.fitness.values[0])
                for ind in population
            ]

            return np.cumsum(fitness_values)


    def _select_parent(self, population, cumulative):
        r = random.uniform(0, cumulative[-1])
        idx = bisect.bisect_left(cumulative, r)
        return population[idx]
    


    def evolve(self, generations=500, resume_from_checkpoint=False):
        """
        Ejecuta el ciclo evolutivo completo.
        Retorna (best_nn, history).
        """
        self._stop_requested = False
        self._history = []

        if resume_from_checkpoint and os.path.exists("models/checkpoint.json"):
            print("Cargando checkpoint...")
            with open("models/checkpoint.json", "r") as f:
                checkpoint = json.load(f)

            # 1. Restaurar history
            self._history = checkpoint.get("history", [])

            # 2. Restaurar best_individual (si existe)
            best_data = checkpoint.get("best_individual")
            if best_data:
                self._best_individual = creator.Individual(best_data)
            else:
                self._best_individual = None

            start_gen = checkpoint["gen"]
            population = []
            for ind_data in checkpoint["population"]:
                ind = creator.Individual(ind_data)
                population.append(ind)
            self._history = checkpoint.get("history", [])
        else:
            start_gen = 0
            population = self._toolbox.population(n=self.pop_size)

        if self._best_individual is not None and self._history:
            self._best_individual.fitness.values = (
                max(
                    s["best_fitness"]
                    for s in self._history
                ),
            )

            best_fitness = self._best_individual.fitness.values[0]

            best_snake_score = max(
                s["global_best_score"] 
                for s in self._history
            )
        else:
            best_fitness = 0
            best_snake_score = 0


        gen = start_gen
        t_start = time.time()

        # ── Pool de procesos (solo si parallel=True) ──────────────────
        pool = None
        if self.parallel:
            try:
                pool = mp.Pool()
            except Exception:
                pool = None

        try:
            while not self._stop_requested and gen < generations:
                gen += 1

                # ── Evaluar toda la población ──────────────────────────
                to_eval = [i for i, ind in enumerate(population)
                           if not ind.fitness.valid]

                if pool and to_eval:
                    # Evaluación paralela
                    args = [list(population[i]) for i in to_eval]
                    results = pool.map(_evaluate_snake_worker, args)
                    for idx, result in zip(to_eval, results):
                        population[idx].fitness.values = (result[0],)
                        population[idx].game_score = result[1]
                        population[idx].game_lifetime = result[2]
                        population[idx].food_list = result[3]
                else:
                    # Evaluación serial
                    for ind in population:
                        if not ind.fitness.valid:
                            ind.fitness.values = self._evaluate_snake(ind)


                 # ── Encontrar el mejor ─────────────────────────────────
                current_best = max(population,
                                   key=lambda x: x.fitness.values[0])
                current_best_fitness = current_best.fitness.values[0]
                current_best_score = getattr(current_best, "game_score", 0)

                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    self._best_individual = self._toolbox.clone(current_best)
                    self._best_individual.game_score = current_best_score
                    self._best_individual.game_lifetime = getattr(
                        current_best, "game_lifetime", 0
                    )
                    self._best_individual.food_list = getattr(
                        current_best, "food_list", []
                    )

                all_scores = [getattr(ind, "game_score", 0) for ind in population]
                best_snake_score = max(best_snake_score, max(all_scores) if all_scores else 0)

                # Guardar snapshot de esta generación
                snapshot = {
                    "gen": gen,
                    "weights": np.array(list(current_best), dtype=np.float64),
                    "score": current_best_score,
                    "fitness": current_best_fitness,
                    "food_list": getattr(current_best, "food_list", []),
                    "lifetime": getattr(current_best, "game_lifetime", 0),
                }
                self.gen_snapshots.append(snapshot)

                # ── Stats ──────────────────────────────────────────────
                all_fitness = [ind.fitness.values[0] for ind in population]
                all_lifetimes = [getattr(ind, "game_lifetime", 0) for ind in population]
                foods_eaten = [max(0, s - 3) for s in all_scores]
                stats = {
                    "gen": gen,
                    "best_fitness": current_best_fitness,
                    "avg_fitness": sum(all_fitness) / len(all_fitness),
                    "best_score": current_best_score,
                    "global_best_score": best_snake_score,
                    "avg_score": sum(all_scores) / len(all_scores),
                    "max_lifetime": max(all_lifetimes) if all_lifetimes else 0,
                    "avg_lifetime": sum(all_lifetimes) / len(all_lifetimes) if all_lifetimes else 0,
                }
                self._history.append(stats)

                # ── ETA ────────────────────────────────────────────────
                elapsed = time.time() - t_start
                eta_sec = (elapsed / gen) * (generations - gen)
                eta_str = (f"{eta_sec / 60:.1f}min"
                           if eta_sec > 60 else f"{eta_sec:.0f}s")

                print(
                    f"Gen {gen:4d} | "
                    f"Best Fitness: {stats['best_fitness']:12.0f} | "
                    f"Score: {stats['best_score']:3d} | "
                    f"HighScore: {stats['global_best_score']:3d} | "
                    f"Avg Score: {stats['avg_score']:.1f} | "
                    f"ETA: {eta_str}"
                )

                # ── Callback para UI ──────────────────────────────────
                if self.on_generation:
                    self.on_generation(gen, stats, self._best_individual)

                # ── Checkpoint Periódico ──────────────────────────────
                if gen % 10 == 0 or gen == generations:
                    os.makedirs("models", exist_ok=True)
                    try:
                        with open("models/checkpoint.json", "w") as f:
                            json.dump({
                                "gen": gen,
                                "history": self._history,
                                "population": [list(ind) for ind in population],
                                "best_individual":
                                    list(self._best_individual)
                                    if self._best_individual
                                    else None,
                            }, f)
                    except Exception as e:
                        print(f"Error guardando checkpoint: {e}")

                # ── Natural Selection (Population.naturalSelection) ───
                new_population = []

                # Elitismo: preservar el mejor con su fitness
                elite_count = 1

                sorted_pop = sorted(
                    population,
                    key=lambda x: x.fitness.values[0],
                    reverse=True
                )

                for elite_src in sorted_pop[:elite_count]:
                    elite = self._toolbox.clone(elite_src)
                    elite.fitness.values = elite_src.fitness.values
                    new_population.append(elite)
                
                cumulative = self._build_roulette(population)

                # Generar el resto por crossover + mutación
                for _ in range(elite_count, self.pop_size):
                    parent1 = self._select_parent(population,cumulative)
                    parent2 = self._select_parent(population,cumulative)

                    # Crossover 
                    child_flat = [
                        p1 if random.random() < 0.5 else p2
                        for p1, p2 in zip(parent1, parent2)
                    ]

                    # Mutación plana (80% pequeña, 20% grande)
                    child_np = np.array(child_flat, dtype=np.float64)
                    mask = np.random.random(self._chrom_size) < self.mutation_rate
                    noise = np.zeros(self._chrom_size)
                    
                    small_mask = (
                        np.random.random(self._chrom_size)
                        < 0.8
                    )
                    
                    noise[small_mask] = (
                        np.random.randn(np.count_nonzero(small_mask))
                        * 0.1
                    )

                    noise[~small_mask] = (
                        np.random.randn(np.count_nonzero(~small_mask))
                        * 0.5
                    )

                    child_np[mask] += noise[mask]

                    # Convertir a individuo DEAP
                    child = creator.Individual(child_np.tolist())
                    new_population.append(child)

                population = new_population

                if self._stop_requested:
                    print(f"\n[STOP]  Evolución detenida en generación {gen}")
                    break

        finally:
            if pool:
                pool.close()
                pool.join()

        # ── Retornar mejor NN ─────────────────────────────────────────
        if getattr(self, "_best_individual", None) is None:
            best_nn = NeuralNetwork()
        else:
            best_nn = self._make_nn(self._best_individual)

        return best_nn, self._history


# ── Ejecución directa (entrenamiento por consola) ────────────────────────────
if __name__ == "__main__":
    mp.freeze_support()

    print("=" * 60)
    print("  SnakeAI — Entrenamiento con Algoritmo Genético (DEAP)")
    print("=" * 60)
    print(f"  Población  : {POP_SIZE}")
    print(f"  Mutación   : {MUTATION_RATE * 100}%")
    nn = NeuralNetwork()
    print(f"  Red        : {nn}")
    print(f"  Cromosoma  : {nn.chromosome_size()} parámetros")
    print(f"  Paralelo   : Sí ({mp.cpu_count()} cores)")
    print("=" * 60 + "\n")

    ga = GeneticAlgorithm(seed=42, parallel=True)
    history = []
    best_nn = NeuralNetwork()

    try:
        best_nn, history = ga.evolve(GENERATIONS)
    except KeyboardInterrupt:
        print("\n[STOP] Detenido por usuario — guardando...")
        if getattr(ga, "_best_individual", None):
            best_nn = ga._make_nn(ga._best_individual)
        history = getattr(ga, "_history", [])
    finally:
        os.makedirs("models", exist_ok=True)
        np.save("models/best_weights.npy", best_nn.get_flat_weights())
        print("✅ Modelo guardado en models/best_weights.npy")
        snapshots_data = []

        for snap in ga.gen_snapshots:
            snapshots_data.append({
                "gen": snap["gen"],
                "weights": snap["weights"].tolist(),
                "score": snap["score"],
                "fitness": snap["fitness"],
                "food_list": snap["food_list"],
                "lifetime": snap["lifetime"],
            })

        with open("models/generation_snapshots.json", "w") as f:
            json.dump(snapshots_data, f)

        print("✅ Snapshots guardados en models/generation_snapshots.json")

    if history:
        print(f"\nMejor fitness : {history[-1]['best_fitness']:.0f}")
        print(f"Mejor score   : {history[-1]['global_best_score']}")

        os.makedirs("experiments", exist_ok=True)
        with open("experiments/history.json", "w") as f:
            json.dump(history, f, indent=2)
        print("Historial guardado en experiments/history.json")
