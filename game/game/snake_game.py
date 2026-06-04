"""
SnakeAI — Motor del juego Snake
=================================
Traducción fiel de Snake.pde + Food.pde del proyecto Processing.

Visión: 8 direcciones × 3 valores = 24 inputs
  - [0]: 1 si hay comida en esa dirección, 0 si no
  - [1]: 1 si hay cuerpo en esa dirección, 0 si no
  - [2]: 1 / distancia a la pared

Acciones: 3 direcciones relativas (Adelante=0, Izquierda=1, Derecha=2)

Vida: 200 movimientos iniciales, +100 al comer (max 500).
"""

import math
import random
from utils.constants import GRID_SIZE, INITIAL_LIFE, LIFE_GAIN, MAX_LIFE

# Direcciones absolutas (en celdas del grid)
UP    = (0, -1)
DOWN  = (0,  1)
LEFT  = (-1, 0)
RIGHT = (1,  0)

# Las 8 direcciones de visión (igual que el original)
DIRECTIONS_8 = [
    (-1,  0),   # left
    (-1, -1),   # up-left
    ( 0, -1),   # up
    ( 1, -1),   # up-right
    ( 1,  0),   # right
    ( 1,  1),   # down-right
    ( 0,  1),   # down
    (-1,  1),   # down-left
]


class SnakeGame:
    """
    Motor del juego Snake idéntico al Processing original.
    Grid de GRID_SIZE × GRID_SIZE celdas.
    """

    def __init__(self, grid_size=GRID_SIZE, replay_food=None):
        self.grid_size = grid_size
        self.replay_food_list = replay_food
        self.replay_food_index = 0
        self.reset()

    def reset(self):
        """Reinicia el juego al estado inicial."""
        cx = self.grid_size // 2
        cy = self.grid_size // 2
        # Cabeza + 2 segmentos de cuerpo debajo (como el original)
        self.head = (cx, cy)
        self.body = [(cx, cy + 1), (cx, cy + 2)]
        self._body_set = set(self.body)
        self.score = 1 + len(self.body)  # original empieza en 3 (1 + 2 body)

        self.x_vel = 0
        self.y_vel = -1  # empieza moviéndose hacia arriba

        self.life_left = INITIAL_LIFE
        self.lifetime = 0
        self.dead = False

        # Lista de comida (para replay)
        if self.replay_food_list:
            self.food = tuple(self.replay_food_list[0])
            self.replay_food_index = 1
        else:
            self.food = self._random_food()

        self.food_list = [self.food]

        self.vision = [0.0] * 26
        self.decision = [0.0] * 3
        return self.get_state()

    # ── Movimiento ───────────────────────────────────────────────────────────

    def move(self):
        """Ejecuta un paso del juego (mover, comer, morir)."""
        if self.dead:
            return

        self.lifetime += 1
        self.life_left -= 1

        # Comer antes de mover (como el original)
        if self.head == self.food:
            self._eat()

        # Mover cuerpo
        self._shift_body()

        # Verificar colisiones
        if self._wall_collide(self.head[0], self.head[1]):
            self.dead = True
        elif self._body_collide(self.head[0], self.head[1]):
            self.dead = True
        elif self.life_left <= 0:
            self.dead = True

    def _eat(self):
        """La serpiente come la comida."""
        self.score += 1
        if self.life_left < MAX_LIFE:
            if self.life_left > MAX_LIFE - LIFE_GAIN:
                self.life_left = MAX_LIFE
            else:
                self.life_left += LIFE_GAIN

        # Añadir segmento al final del cuerpo
        if self.body:
            self.body.append(self.body[-1])
        else:
            self.body.append(self.head)
        self._body_set = set(self.body)

        # Nueva comida
        if self.replay_food_list and self.replay_food_index < len(self.replay_food_list):
            self.food = tuple(self.replay_food_list[self.replay_food_index])
            self.replay_food_index += 1
        else:
            self.food = self._random_food()
            while self.food in self._body_set or self.food == self.head:
                self.food = self._random_food()

        self.food_list.append(self.food)

    def _shift_body(self):
        """Mueve el cuerpo siguiendo a la cabeza."""
        old_head = self.head
        self.head = (self.head[0] + self.x_vel, self.head[1] + self.y_vel)

        if self.body:
            prev = old_head
            for i in range(len(self.body)):
                prev, self.body[i] = self.body[i], prev
            self._body_set = set(self.body)

    # ── Colisiones ───────────────────────────────────────────────────────────

    def _body_collide(self, x, y):
        """Verifica colisión con el cuerpo (O(1) con set)."""
        return (x, y) in self._body_set

    def _wall_collide(self, x, y):
        """Verifica colisión con las paredes."""
        return x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size

    def _food_collide(self, x, y):
        """Verifica colisión con la comida."""
        return x == self.food[0] and y == self.food[1]

    # ── Visión (8 direcciones × 3 = 24 inputs) ──────────────────────────────

    def look(self):
        """Mira en 8 direcciones relativas y añade distancias relativas."""
        self.vision = [0.0] * 26
        
        # Determinar el vector forward
        forward = (self.x_vel, self.y_vel)
        if forward == (0, 0):
            forward = (0, -1)
            
        # Rotar DIRECTIONS_8 para que el índice 0 sea siempre 'forward'
        try:
            fwd_idx = DIRECTIONS_8.index(forward)
        except ValueError:
            fwd_idx = 2  # up (default)
            
        rel_directions = DIRECTIONS_8[fwd_idx:] + DIRECTIONS_8[:fwd_idx]

        for i, direction in enumerate(rel_directions):
            temp = self._look_in_direction(direction)
            self.vision[i * 3]     = temp[0]  # 1/dist_food
            self.vision[i * 3 + 1] = temp[1]  # 1/dist_body
            self.vision[i * 3 + 2] = temp[2]  # 1/dist_wall
            
        # Añadir vector de posición relativa a la comida (normalizado)
        dx = (self.food[0] - self.head[0]) / self.grid_size
        dy = (self.food[1] - self.head[1]) / self.grid_size

        # Componentes relativas a la orientación actual
        if self.x_vel == 0 and self.y_vel == -1:      # Arriba
            food_forward = -dy
            food_side = dx
        elif self.x_vel == 1 and self.y_vel == 0:     # Derecha
            food_forward = dx
            food_side = dy
        elif self.x_vel == 0 and self.y_vel == 1:     # Abajo
            food_forward = dy
            food_side = -dx
        else:                                         # Izquierda
            food_forward = -dx
            food_side = -dy

        self.vision[24] = food_forward
        self.vision[25] = food_side

    def _look_in_direction(self, direction):
        """Mira en una dirección. Retorna [1/dist_food, 1/dist_body, 1/dist_wall]."""
        look = [0.0, 0.0, 0.0]
        pos_x, pos_y = self.head
        distance = 0
        food_found = False
        body_found = False

        pos_x += direction[0]
        pos_y += direction[1]
        distance += 1

        while not self._wall_collide(pos_x, pos_y):
            if not food_found and self._food_collide(pos_x, pos_y):
                food_found = True
                look[0] = 1.0 / distance
            if not body_found and self._body_collide(pos_x, pos_y):
                body_found = True
                look[1] = 1.0 / distance

            pos_x += direction[0]
            pos_y += direction[1]
            distance += 1

        look[2] = 1.0 / distance
        return look

    def get_state(self):
        """Retorna el vector de visión (24 floats)."""
        self.look()
        return list(self.vision)

    # ── Direcciones ──────────────────────────────────────────────────────────

    def move_up(self):
        if self.y_vel != 1:  # no ir en dirección opuesta
            self.x_vel = 0
            self.y_vel = -1

    def move_down(self):
        if self.y_vel != -1:
            self.x_vel = 0
            self.y_vel = 1

    def move_left(self):
        if self.x_vel != 1:
            self.x_vel = -1
            self.y_vel = 0

    def move_right(self):
        if self.x_vel != -1:
            self.x_vel = 1
            self.y_vel = 0

    def set_absolute_direction(self, action_index):
        """Modo manual: Arriba=0, Abajo=1, Izquierda=2, Derecha=3"""
        if action_index == 0:
            self.move_up()
        elif action_index == 1:
            self.move_down()
        elif action_index == 2:
            self.move_left()
        elif action_index == 3:
            self.move_right()

    def set_direction(self, action_index):
        """IA: Direcciones relativas. 0=Adelante, 1=Izquierda, 2=Derecha"""
        if action_index == 0:
            pass # Adelante (no cambiar)
        elif action_index == 1:
            # Girar izquierda
            if self.x_vel == 0 and self.y_vel == -1:    # Arriba -> Izquierda
                self.x_vel, self.y_vel = -1, 0
            elif self.x_vel == -1 and self.y_vel == 0:  # Izquierda -> Abajo
                self.x_vel, self.y_vel = 0, 1
            elif self.x_vel == 0 and self.y_vel == 1:   # Abajo -> Derecha
                self.x_vel, self.y_vel = 1, 0
            elif self.x_vel == 1 and self.y_vel == 0:   # Derecha -> Arriba
                self.x_vel, self.y_vel = 0, -1
        elif action_index == 2:
            # Girar derecha
            if self.x_vel == 0 and self.y_vel == -1:    # Arriba -> Derecha
                self.x_vel, self.y_vel = 1, 0
            elif self.x_vel == 1 and self.y_vel == 0:   # Derecha -> Abajo
                self.x_vel, self.y_vel = 0, 1
            elif self.x_vel == 0 and self.y_vel == 1:   # Abajo -> Izquierda
                self.x_vel, self.y_vel = -1, 0
            elif self.x_vel == -1 and self.y_vel == 0:  # Izquierda -> Arriba
                self.x_vel, self.y_vel = 0, -1

    # ── Fitness (idéntico al original) ───────────────────────────────────────

    def calculate_fitness(self):
        if self.score < 10:
            self.fitness = (self.lifetime ** 2) * (2 ** self.score)
        else:
            self.fitness = (self.lifetime ** 2)
            self.fitness *= (2 ** 10)
            self.fitness *= (self.score - 9)
        return self.fitness

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _dist_to_food(self, pos):
        """Distancia euclidiana a la comida."""
        return math.dist(pos, self.food)

    def _random_food(self):
        """Genera una posición aleatoria de comida dentro del grid."""
        x = random.randint(0, self.grid_size - 1)
        y = random.randint(0, self.grid_size - 1)
        return (x, y)

    def get_all_positions(self):
        """Retorna (head, body, food) para renderizado."""
        return self.head, list(self.body), self.food
