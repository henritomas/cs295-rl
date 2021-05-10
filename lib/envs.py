import sys
import os
if "./" not in sys.path:
  sys.path.append("./") 

import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import itertools
from collections import defaultdict
from lib import plotting
from tqdm import tqdm
from copy import deepcopy

matplotlib.style.use('ggplot')

from gym import error, spaces, utils

class CliffWalkingEnv(gym.Env):

    def __init__(self, rows=4, cols=12):

        assert(rows > 1), 'Unsolvable Cliff. Argument rows must be > 1'
        assert(cols > 2), 'No Cliff. Argument cols must be > 2'

        self.grid_size = (rows,cols)
        self.start = [0,0]
        self.goal = [0, cols-1]

        # initial position
        self.position = deepcopy(self.start)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.prod(self.grid_size))

    def get_state(self):
        '''
        Returns the state # based on the agent's position.
        E.g. given a (4,12) cliff with (0,0) origin and an agent at (1,3), 
        state # would be:

        state = (1 * 12) + 3 = 16
        '''

        position = self.position
        cols = self.grid_size[1]

        return (position[0] * cols) + position[1]

    def get_state_onehot(self):

        n_states = np.prod(self.grid_size)

        s = self.get_state()
        s_onehot = np.zeros(np.prod(self.grid_size)).astype(np.float32)
        s_onehot[s] = 1.0

        return s_onehot

    def reset(self):
        '''
        Resets position of agent to (0,0)
        '''

        self.position = deepcopy(self.start)


    def step(self, action):
        '''
        One timestep of the agent performing an action. Actions should be 
        up, down, left, right (integers 0 to 3) limited by the grid. 
        '''

        assert(0 <= action <= 3), "Invalid action. Action should be within [0,3]"

        position = deepcopy(self.position)
        rows, cols = self.grid_size
        
        if action == 0: #up
            position[0] = min(position[0]+1, rows-1)
        elif action == 1: #down
            position[0] = max(position[0]-1, 0)
        elif action == 2: #left
            position[1] = max(position[1]-1, 0)
        elif action == 3: #right
            position[1] = min(position[1]+1, cols-1)

        self.position = position
        new_state = self.get_state()

        # Cliff states are from [1, cols-2], recall 0 is start and cols-1 is goal
        if 0 < new_state < cols-1:
            reward = -100
        else:
            reward = -1

        # Did the agent fall off the cliff or reach the goal? 
        done = (0 < new_state <= cols-1)


        return new_state, reward, done, {}        