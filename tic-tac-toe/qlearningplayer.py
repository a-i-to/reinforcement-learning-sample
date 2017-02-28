"""
Q-learning Player
"""
import itertools
from collections import defaultdict
from copy import copy

import numpy as np

import player


class QLearningPlayer(player.Player):
    """ Q-learning player.

    Args:
        lr (float) : learning rate
    """

    def __init__(self, random_state, lr):
        super().__init__(random_state)
        self.lr = lr

    def fit(self, policy_func, opponent, discount, m):
        """ learning q function.

        Args:
            policy_func (:obj): policy function (callable)
            opponent (:obj):
            discount (float): discount rate
            m (int): num of iteretions
            seed (int): random seed

        Returns:
            self (:obj):
        """
        self.reset()
        for j in range(m):
            state = 0
            if self.random_state.rand() < 0.5:
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
                    self.update_q(encoded, action, reward, next_state,
                                  discount, True)
                    break

                next_state = self.decode(encode_idx, next_state)
                opponent_action = opponent.move(next_state)
                next_next_state, reward, is_finished \
                    = self.update_state(next_state, False, opponent_action)
                encode_idx, next_next_encoded \
                    = self.encode(next_next_state)

                self.update_q(encoded, action, reward, next_next_encoded,
                              discount, is_finished)

                if is_finished:
                    break

                state = self.decode(encode_idx, next_next_encoded)
                if state == 101136:
                    print(next_state)
                    print(opponent_action)
        return self

    def update_q(self, state, action, reward, next_state, discount,
                 is_finished):
        if is_finished:
            self.q[state][action] += self.lr * \
                                     (reward - self.q[state][action])
        else:
            self.q[state][action] += self.lr * \
                                     (reward - self.q[state][action] +
                                      discount *
                                      max(self.q[next_state]))
