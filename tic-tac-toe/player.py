"""
 Tic-Tac-Toe Abstract Player
"""
import itertools
from collections import defaultdict

import numpy as np

import game


class Player:
    """ Players' superclass.
    """
    T = 5
    REWARD_WIN = 5
    REWARD_LOSE = -5
    REWARD_DRAW = 0
    REWARD_ONGOING = 0

    ROT_MATRIX = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [6, 3, 0, 7, 4, 1, 8, 5, 2],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
        [2, 5, 8, 1, 4, 7, 0, 3, 6],
        [2, 1, 0, 5, 4, 3, 8, 7, 6],
        [0, 3, 6, 1, 4, 7, 2, 5, 8],
        [6, 7, 8, 3, 4, 5, 0, 1, 2],
        [8, 5, 2, 7, 4, 1, 6, 3, 0]
    ]

    def __init__(self, random_state):
        self.random_state = random_state
        self.reset()

    def reset(self):
        self.q = defaultdict(lambda: [0] * game.NUM_CELLS)
        self.game_result = [0, 0, 0]

    def take_action(self, policy):
        """ select an action.

        Args:
            policy (List[float]):
        """
        accum = list(itertools.accumulate(policy))
        rand_value = self.random_state.rand()
        return [i for i, v in enumerate(accum) if v >= rand_value][0]

    def update_state(self, state, is_my_move, action):
        next_state = self.generate_next(state, is_my_move, action)
        game_state = self.is_finished(next_state)
        if game_state == 'PLAYER_WIN':
            reward = self.REWARD_WIN
            self.game_result[0] += 1
        elif game_state == 'OPPONENT_WIN':
            reward = self.REWARD_LOSE
            self.game_result[1] += 1
        elif game_state == 'FINISH':
            reward = self.REWARD_DRAW
            self.game_result[2] += 1
        else:
            reward = self.REWARD_ONGOING
        return (next_state, reward, game_state != 'CONTINUE')

    def generate_next(self, state, is_my_move, action):
        if is_my_move:
            return state | (1 << (action * 2))
        else:
            return state | (1 << ((action * 2) + 1))

    def is_finished(self, state):
        for i in range(game.NUM_CELLS_IN_ROW):
            row_state = (state & (63 << i * 6)) >> i * 6
            if row_state == 21:
                return "PLAYER_WIN"
            elif row_state == 42:
                return "OPPONENT_WIN"
            col_state = (state & (12483 << i * 2)) >> i * 2
            if col_state == 4161:
                return "PLAYER_WIN"
            elif col_state == 8322:
                return "OPPONENT_WIN"
        diag_state = state & 197379
        if diag_state == 65793:
            return "PLAYER_WIN"
        elif diag_state == 131586:
            return "OPPONENT_WIN"
        diag_state2 = state & 13104
        if diag_state2 == 4368:
            return "PLAYER_WIN"
        elif diag_state2 == 8736:
            return "OPPONENT_WIN"
        if len([i for i in range(game.NUM_CELLS)
                if (3 << i * 2) & state == 0]) == 0:
            return "FINISH"
        else:
            return "CONTINUE"

    def encode(self, state):
        as_array = np.zeros(game.NUM_CELLS, int)
        for i in range(game.NUM_CELLS):
            as_array[i] = (state >> i * 2) & 3

        min = state
        min_index = 0
        for i, indice in enumerate(self.ROT_MATRIX[1:]):
            arr = as_array[indice]
            val = 0
            for j in range(game.NUM_CELLS):
                val |= (arr[j] << j * 2)
            if val < min:
                min = val
                min_index = i + 1
        return (min_index, min)

    def decode(self, index, val):
        if index == 0:
            return val

        as_array = np.zeros(game.NUM_CELLS, int)
        for i in range(game.NUM_CELLS):
            as_array[i] = (val >> i * 2) & 3

        arr = as_array[self.ROT_MATRIX[index]]
        val = 0
        for i in range(game.NUM_CELLS):
            val |= arr[i] << i * 2
        return val
