"""
SARSA Player
"""
import itertools
from collections import defaultdict
from copy import copy

import numpy as np

import player


class SarsaPlayer(player.Player):
    """ SARSA player.
    """

    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def fit(self, policy_func, opponent, discount, l, m, seed=None):
        """ learning q values.
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
                    policy = policy_func(self.q, encoded)
                    action = self.take_action(policy)
                    next_state, reward, is_finished \
                        = self.update_state(encoded, True, action)
                    if is_finished:
                        episode.append((encoded, action, reward, next_state))
                        break

                    next_state = self.decode(encode_idx, next_state)
                    opponent_action = opponent.move(next_state)
                    next_next_state, reward, is_finished \
                        = self.update_state(next_state, False, opponent_action)
                    encode_idx, next_next_encoded \
                        = self.encode(next_next_state)

                    episode.append((encoded, action, reward,
                                    next_next_encoded))
                    if is_finished:
                        break

                    state = self.decode(encode_idx, next_next_encoded)

                self.update_new_q(new_q, episode, discount)

            self.update_q(new_q)

        return self

    def update_new_q(self, new_q, episode, discount):
        episode_len = len(episode)

        for i, v in enumerate(reversed(episode)):
            state = v[0]
            action = v[1]
            reward = v[2]

            if i == 0:
                new_q[state][action] += self.lr * \
                                        (reward - new_q[state][action])
            else:
                next_data = episode[episode_len - i]
                next_state = next_data[0]
                next_action = next_data[1]
                new_q[state][action] += self.lr * \
                    (reward - new_q[state][action] +
                     discount *
                     new_q[next_state][next_action])

    def update_q(self, new_q):
        self.q = new_q
