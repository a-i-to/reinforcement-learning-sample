"""
SARSA Player 
"""

import numpy as np
import itertools
from collections import defaultdict
from copy import copy

class SarsaPlayer:
    """ SARSA player.

    Args:
        policy_func (function):  policy generator.
        discount (float): discount rate.
        lr (float) : learnning rate

    Attributes:
        T (int): max steps in each episode.
        NUM_STATES (int): num of states
        NUM_ACTIONS (int): num of actions
        q : a lookup table representing q function.
    """
    T = 5
    NUM_ACTIONS = 9

    ROT_MATRIX = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [6, 3, 0, 7, 4, 1, 8, 5, 2],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
        [2, 5, 8, 1, 4, 7, 0, 3, 6],
        [2, 1, 0, 5, 4, 3, 8, 7, 6],
        [0, 3, 6, 1, 4, 7, 2, 5, 8],
        [6, 7 ,8, 3, 4, 5, 0, 1, 2],
        [8, 5, 2, 7, 4, 1, 6, 3, 0]
    ]
    
    def __init__(self, policy_func, discount, lr):
        self._policy_func = policy_func
        self._discount = discount
        self._lr = lr
        self.q = defaultdict(lambda: [0] * self.NUM_ACTIONS)
        
    def fit(self, opponent, l, m, seed=None):
        """ learning q function.
        
        Args:
            opponent (:obj): 
            l (int): num of iterations
            m (int): num of episodes

        Returns:
            self (:obj): 
        """
        np.random.seed(seed)
        opponent.random = np.random
        
        for i in range(l):
            new_q = copy(self.q)
            for j in range(m):
                episode = []

                state = 0
                if np.random.rand() < 0.5:
                    action = opponent.move(state)
                    next_state = self.update_state(state, False, action)[0]
                    state = next_state

                for k in range(self.T):
                    encode_idx, encoded = self.encode(state)
                    policy = self._policy_func.execute(self.q, encoded)
                    action = self.take_action(policy)
                    next_state, reward, is_finished \
                        = self.update_state(encoded, True, action)
                    if is_finished:
                        episode.append((encoded, action, reward, next_state))
                        break

                    next_state = self.decode(encode_idx, next_state)
                    opponent_action = opponent.move(next_state)
                    next_next_state, _, is_finished \
                        = self.update_state(next_state, False, opponent_action)
                    encode_idx, next_next_encoded \
                        = self.encode(next_next_state)

                    episode.append((encoded, action, reward,
                                    next_next_encoded))
                    if is_finished:
                        break

                    state = self.decode(encode_idx, next_next_encoded)

                self.update_new_q(new_q, episode)

            self.update_q(new_q)

        return self
        
    def take_action(self, policy):
        """ select an action.
        
        Args:
            policy (List[float]): 
        """
        accum = list(itertools.accumulate(policy))
        rand_value = np.random.rand()
        return [i for i,v in enumerate(accum) if rand_value <= v][0]

    def update_state(self, state, is_my_move, action):
        next_state = self.generate_next(state, is_my_move, action)
        game_state = self.is_finished(next_state)
        if game_state == 'PLAYER_WIN':
            reward = 5
        else:
            reward = 0
        return (next_state, reward, game_state != 'CONTINUE')
        
    def generate_next(self, state, is_my_move, action):
        if is_my_move:
            return state | (1 << (action * 2))
        else:
            return state | (1 << ((action * 2) + 1))
        
    def is_finished(self, state):

        for i in range(3):
            row_state = (state & (63 << i * 6)) >> i * 6
            if row_state == 21:
                return "PLAYER_WIN"
            elif row_state == 42:
                return "OPPONENT_WIN"
            col_state = (state & 12483 << i * 2) >> i * 2
            if col_state == 4161:
                return "PLAYER_WIN"
            elif col_state == 8332:
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
        if len([i for i in range(self.NUM_ACTIONS)
                    if (3 << i * 2) & state == 0]) == 0:
            return "FINISHED"
        else:
            return "CONTINUE"

    def update_new_q(self, new_q, episode):
        episode_len = len(episode)
        
        for i, v in enumerate(reversed(episode)):
            state = v[0]
            action = v[1]
            reward = v[2]

            if i == 0:
                new_q[state][action] += self._lr * \
                                        (reward - new_q[state][action])
            else:
                next_data = episode[episode_len - i]
                next_state = next_data[0]
                next_action = next_data[1]
                new_q[state][action] += self._lr * \
                                        (reward - new_q[state][action] +
                                         self._discount *
                                         new_q[next_state][next_action])

    def update_q(self, new_q):
        self.q = new_q

    def encode(self, state):
        as_array = np.zeros(9, int)
        for i in range(9):
            as_array[i] = (state >> i * 2) & 3

        min = state
        min_index = 0
        for i, indice in enumerate(self.__class__.ROT_MATRIX[1:]):
            arr = as_array[indice]
            val = 0
            for j in range(9):
                val |= (arr[j] << j * 2)
            if val < min:
                min = val
                min_index = i + 1
        return (min_index, min)

    def decode(self, index, val):
        if index == 0:
            return val
        
        as_array = np.zeros(9, int)
        for i in range(9):
            as_array[i] = (val >> i * 2) & 3
        
        arr = as_array[self.__class__.ROT_MATRIX[index]]
        val = 0
        for i in range(9):
            val |= arr[i] << i * 2
        return val

