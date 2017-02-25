"""
Monte Calro Player
"""
import itertools
from collections import defaultdict

import numpy as np

import game
import player


class MonteCalroPlayer(player.Player):
    """ Monte Calro player.
    """

    def fit(self, policy_func, opponent, discount, l, m):
        """ learning q function.

        Args:
            policy_func (:obj): policy function (callable)
            opponent (:obj):
            discount (float): discount rate
            l (int): num of iterations
            m (int): num of episodes

        Returns:
            self (:obj):
        """
        self.reset()
        for i in range(l):
            sampling_data = defaultdict(lambda: (0, 0))
            for j in range(m):
                episode = []

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

                self.update_sampling_data(sampling_data, episode, discount)

            self.update_q(sampling_data)

        return self

    def update_sampling_data(self, sampling_data, episode, discount):
        future_reward = 0
        for i, v in enumerate(reversed(episode)):
            state = v[0]
            action = v[1]
            reward = v[2] + discount * future_reward
            future_reward = reward

            key = (state, action)
            value = sampling_data[key]
            sampling_data[key] = (value[0] + reward,
                                  value[1] + 1)

    def update_q(self, sampling_data):
        self.q = defaultdict(lambda: [0] * game.NUM_CELLS)
        for k, v in sampling_data.items():
            self.q[k[0]][k[1]] = v[0] / v[1]
