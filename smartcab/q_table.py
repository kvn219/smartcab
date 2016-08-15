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


# Alphas
def decay1(t):
    return 1 / (t + 1)


def decay2(t):
    return 1 / (t + 2)


def decay4(t):
    """TODO: Explore 1 - (1/sqrt(t+x) as a parameter for gamma"""
    return 1 - (1/np.sqrt(t + 3))
    # return 1 - (1/np.sqrt(t))
    # return 1 - (1/np.log(t+3))
    # return 1-(t/np.exp(t + 10))


def constant(t):
    return 0.1


# Epsilons 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy(t):
    return -np.log(sigmoid(t + 2))


def arithmetic_average(t):
    """Learning Rate for Stationary"""
    if t == 0:
        return 1
    else:
        return 1 / t


def detect_function(f):
    try:
        if f is cross_entropy:
            return "Cross Entropy"
        elif f is constant:
            return "One"
    except:
        return "Unknown"
