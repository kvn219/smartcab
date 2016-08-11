from __future__ import division
from collections import defaultdict
from itertools import product
import random
import numpy as np


def generate_q_table(initial_value):
    way_points = ['left', 'right', 'forward']
    oncoming = [None, 'forward', 'left', 'right']
    actions = [None, 'forward', 'left', 'right']
    lights = ['red', 'green']
    left = [None, 'forward', 'left', 'right']
    state = [lights, oncoming, left, way_points]
    q_table = defaultdict(tuple)
    for state in product(*state):
        if initial_value == 'random':
            q_table[state] = {action: np.random.choice(10) for action in actions}
        elif initial_value == 'normal':
            q_table[state] = {action: np.random.randn() for action in actions}
        elif initial_value == 'zero':
            q_table[state] = {action: 0 for action in actions}
        elif initial_value == 'one':
            q_table[state] = {action: 1 for action in actions}
        elif initial_value == 'hundred':
            q_table[state] = {action: 100 for action in actions}
        else:
            print("{}".format(0))
            q_table[state] = {action: np.random.uniform(0.0, 0.4) for action in actions}
    return q_table


def best_choice(state):
    max_value = max(state.values())
    best_actions = [action for action, reward in state.items() if reward == max_value]
    choice = random.choice(best_actions)
    return choice


def quadratic(t):
    return t ** 2


def neg_log(t):
    return 1 - np.log(t + 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy(t):
    return -np.log(sigmoid(t + 2))


def decay(t):
    return 1 / np.log(t + 2)


def default(t):
    return 1.0


def detect_function(f):
    if f is cross_entropy:
        return "Cross Entropy"
    elif f is standard_prob:
        return "Standard"
    elif f is default:
        return "One"
