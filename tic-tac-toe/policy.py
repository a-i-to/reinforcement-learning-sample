"""
Policy functions
"""

import numpy as np
import math

NUM_ACTIONS = 9

def can_take(state, action):
    return ((3 << action * 2) & state) == 0


class Greedy:

    def __call__(self, q, state):
        policy = [0] * NUM_ACTIONS
        pairs = [(i,v) for i, v in enumerate(q[state]) if can_take(state, i)]
        policy[max(pairs, key=lambda x: x[1])[0]] = 1
        return policy


class Softmax:

    def __init__(self, temperature):
       self._temperature = temperature

    def __call__(self, q, state):
        policy = [0] * NUM_ACTIONS
        pairs = [(i,v) for i, v in enumerate(q[state]) if can_take(state, i)]
        for p in pairs:
            policy[p[0]] = math.exp(p[1] / self._temperature)
        policy_sum = sum(policy)
        return [v / policy_sum for v in policy]


class EGreedy:

    def __init__(self, epsilon):
        self._epsilon = epsilon

    def __call__(self, q, state):
        policy = [0] * NUM_ACTIONS
        pairs = [(i,v) for i, v in enumerate(q[state]) if can_take(state, i)]
        for p in pairs:
            policy[p[0]] = self._epsilon / len(pairs)
        policy[max(pairs, key=lambda x: x[1])[0]] += (1 - self._epsilon)
        return policy
