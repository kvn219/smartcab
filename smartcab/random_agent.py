from __future__ import division
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from q_table import *
import numpy as np
import pandas as pd
import random
import os


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.q_table = generate_q_table(self.env.initial_value)
        self.actions = [None, 'forward', 'left', 'right']
        self.trial = 0
        self.max_moves = 0
        self.time = 0.0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.max_moves = 0

    def random_action(self):
        return random.choice(self.actions)

    def keep_track(self):
        if self.planner.next_waypoint() is None:
            reached_destination.append(1)
        else:
            reached_destination.append(0)

    def arg_max(self):
        max_reward = self.q_table[self.state][best_choice(self.q_table[self.state])]
        return max_reward

    def q_learn(self, s1, s2, t):
        learning_rate = self.env.alpha(t)
        discount_rate = self.env.gamma
        q_val = s1 + learning_rate * (s2 + discount_rate * self.arg_max() - s1)
        return q_val

    def informed_agent(self, inputs):
        way_point = self.next_waypoint # way point

        if way_point == 'forward':
            if inputs['light'] == 'green':
                return way_point
        elif way_point == 'right':
            if inputs['light'] == 'green':
                return way_point
            elif inputs['light'] == 'red' and inputs['oncoming'] is None:
                return way_point
        elif way_point == 'left':
            if inputs['light'] == 'green' and inputs['oncoming'] != 'left':
                return way_point
        else:
            return None

    def record_trial(self, action, reward, inputs, t, deadline):
        result = dict()
        result['action'] = action
        result['reward'] = reward
        result['t'] = t
        result['light'] = inputs['light']
        result['oncoming'] = inputs['oncoming']
        result['location'] = self.env.agent_states[self]["location"]
        result['success'] = self.planner.next_waypoint() is None
        result['destination'] = self.env.agent_states[self]["destination"]
        result['moves_taken'] = self.max_moves - deadline
        result['trip'] = self.trial
        result['alpha'] = self.env.alpha
        result['gamma'] = self.env.gamma
        result['initial_value'] = self.env.initial_value
        result['sim_num'] = self.env.sim_num
        result['deadline'] = self.max_moves
        return result

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        if t is 0:
            self.max_moves = deadline
            self.trial += 1
        self.time += 1.0

        # Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # Select action according to policy
        action = random.choice(self.actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Keep track of results
        result = self.record_trial(action, reward, inputs, t, deadline)
        results.append(result)

        # Keep track of reaching destination
        self.keep_track()

        # Learn policy based on state, action, reward
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        s1 = self.q_table[self.state][action]
        s2 = self.env.act(self, action)
        self.q_table[self.state][action] = self.q_learn(s1, s2, self.time)


def run(alpha=default, gamma=1.0, epsilon=default, initial_value='zero', sim_num=1):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(alpha, gamma, epsilon, initial_value, sim_num)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)

    sim.run(n_trials=100)  # run for a specified number of trials
    print(np.sum(reached_destination))

if __name__ == '__main__':
    reached_destination = []
    results = []
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, '../results/random_agent.json')
    run()
    pd.DataFrame(results).to_json(path)
    print(path)
