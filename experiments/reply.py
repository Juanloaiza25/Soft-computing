import json
import numpy as np

import sys
import os

# Añadir el directorio raíz al path para poder importar 'ai'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ai.neural_network import NeuralNetwork
from game.snake_game import SnakeGame
from utils.constants import GRID_SIZE


def load_generation(gen_number):
    with open("models/generation_snapshots.json", "r") as f:
        snapshots = json.load(f)

    snapshot = snapshots[gen_number - 1]

    nn = NeuralNetwork()
    nn.set_flat_weights(
        np.array(snapshot["weights"], dtype=np.float64)
    )

    return nn, snapshot

nn, info = load_generation(50)

game = SnakeGame(GRID_SIZE, replay_food=info["food_list"])

while not game.dead:
    game.look()

    decision = nn.output(game.vision)

    action = int(np.argmax(decision))

    game.set_direction(action)
    game.move()

print("Score final:", game.score)
print(info["food_list"][:10])
print(len(info["food_list"]))