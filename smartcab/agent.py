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
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)
        # Initialize any additional variables here
        self.q_table = generate_q_table(self.env.initial_value)
        self.actions = [None, 'forward', 'left', 'right']
        self.trial = 0.0
        self.max_moves = 0.0
        self.time = 0.0
        self.inital_location = 0.0
        self.t_q_val = 0.0

    def reset(self, destination=None):
        self.planner.route_to(destination)

        # Prepare for a new trip; reset any variables here, if required
        self.max_moves = 0.0
        self.time = 0.0

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
        discount_rate = self.env.gamma(t)
        q_val = s1 + learning_rate * (s2 + discount_rate * self.arg_max() - s1)
        return q_val

    def record_trial(self, action, reward, inputs, t, deadline):
        result = dict()
        result['action'] = action
        result['reward'] = reward
        result['t'] = t
        result['light'] = inputs['light']
        result['oncoming'] = inputs['oncoming']
        result['location'] = self.env.agent_states[self]["location"]
        result['success'] = self.planner.next_waypoint() is None
        result['way_point'] = self.planner.next_waypoint()
        result['destination'] = self.env.agent_states[self]["destination"]
        result['moves_taken'] = self.max_moves - deadline
        result['trip'] = self.trial
        result['alpha'] = self.env.alpha
        result['gamma'] = self.env.gamma
        result['epsilon'] = self.env.epsilon
        result['initial_value'] = self.env.initial_value
        result['sim_num'] = self.env.sim_num
        result['max_moves'] = self.max_moves
        result['inital_location'] = self.inital_location
        result['q_val'] = self.t_q_val
        result['inputs'] = inputs
        result['left'] = inputs['left']
        return result

    def update(self, t):

        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Keep track of moves, trials, and time
        if t is 0:
            self.max_moves = deadline
            self.trial += 1
            self.inital_location = self.env.agent_states[self]["location"]
        self.time += 1.0

        # Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # Probability of selecting action
        prob = self.env.epsilon(t)

        # Select action according to your policy
        self.t_q_val = self.q_table[self.state]
        action = (np.random.choice([
            best_choice(self.q_table[self.state]), self.random_action()],
            1, p=[1 - prob, prob])[0])

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Keep track of results
        result = self.record_trial(action, reward, inputs, t, deadline)
        results.append(result)
        self.keep_track()

        # Learn policy based on state, action, reward
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        s1 = self.q_table[self.state][action]
        s2 = self.env.act(self, action)
        self.q_table[self.state][action] = self.q_learn(s1, s2, self.time)


def run(alpha=constant, gamma=constant, epsilon=constant, initial_value='random', sim_num=1):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(alpha, gamma, epsilon, initial_value, sim_num)

    # create agent
    a = e.create_agent(LearningAgent)

    # specify agent to track
    e.set_primary_agent(a, enforce_deadline=True)

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)
    sim.run(n_trials=100)

    # Print # of successful trips
    print(np.sum(reached_destination))

if __name__ == '__main__':
    reached_destination = []
    results = []
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, '../results/optimal_agent.json')

    # Run below to run optimal agent
    run(alpha=decay2, gamma=constant, epsilon=cross_entropy, initial_value='zero', sim_num=1)

    # Save results to json file
    pd.DataFrame(results).to_json(path)
    print(path)
