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

matplotlib.style.use('ggplot')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EgreedyPolicy():
    '''
    Creates an egreedy policy object, list of probabilities for each action
    in the form of a numpy array.
    '''

    def __init__(self, Q, epsilon, nA):
        '''
        Args:
            Q: Dictionary that maps state -> action values. Each value is an np 
            array of length nA.
            epsilon: Probability that model chooses a random action.
            nA: Number of actions in the environment. (no. of Actions)
        '''

        self.Q = Q
        self.epsilon = epsilon
        self.nA = nA

    def step(self, state):
        '''
        Returns 1D numpy array of action probabilities when given state.
        e.g. if nA = 3, and best action is #3, 
        out would be [e/(nA-1), e/(nA-1), 1-e]
        '''
        Q, epsilon, nA = self.Q, self.epsilon, self.nA

        pA = np.ones(nA) * (epsilon/(nA-1))
        bestA = np.argmax(Q[state])
        pA[bestA] = (1 - epsilon)

        # Choose action greedily(1-epsilon)
        a = np.random.choice(np.arange(nA), p=pA)

        return a

def q_learning(env, n_episodes, e_initial, e_final, e_decay, alpha=0.1, gamma=1.0):
    '''
    Implementation of the Q-Learning algorithm: Off-policy TD control.
    Follows an epsilon-greedy policy with parameters set by arguments.

    Args:
        env: OpenAI gym environment
        n_episodes: number of training eps
        alpha: learning rate for Q update
        gamma: discount factor for future reward estimates

    Returns:
        (Q, stats): Tuple of new state-action dictionary and training stats
    '''

    # number of Actions in env
    nA = env.action_space.n

    # initialize Q(S,A) : state-action -> action-value function
    Q = defaultdict(lambda: np.zeros(nA))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(n_episodes),
        episode_rewards=np.zeros(n_episodes))
    
    epsilon = e_initial
    for episode in tqdm(range(n_episodes)):

        #Make/update the policy
        policy = EgreedyPolicy(Q, epsilon, nA)

        # init total rewards per episode for statistics
        total_r = 0

        # Initilize environment, Get observation s
        s = env.reset()

        # Decaying epsilon
        if epsilon > e_final:
            epsilon *= e_decay

        for t in itertools.count(): # Counts timesteps t for each episode
            
            # Get action given state s, choose action egreedily
            a = policy.step(s)

            # Perform action, observe R, S'
            s_prime, r, done, _ = env.step(a)

            # Q-Update
            Q[s][a] = Q[s][a] + alpha * (r + (gamma * np.max(Q[s_prime])) - Q[s][a])

            # state update
            s = s_prime

            total_r += r
            if done: 
                break

        stats.episode_lengths[episode] = t
        stats.episode_rewards[episode] = total_r
            
    return Q, stats

def q_learning_frozenlake(env, n_episodes, e_initial, e_final, e_decay, alpha=0.1, gamma=1.0):
    '''
    Implementation of the Q-Learning algorithm: Off-policy TD control.
    Follows an epsilon-greedy policy with parameters set by arguments.

    Args:
        env: OpenAI gym environment
        n_episodes: number of training eps
        alpha: learning rate for Q update
        gamma: discount factor for future reward estimates

    Returns:
        (Q, stats): Tuple of new state-action dictionary and training stats
    '''

    # number of Actions in env
    nA = env.action_space.n

    # initialize Q(S,A) : state-action -> action-value function
    Q = defaultdict(lambda: np.zeros(nA))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(n_episodes),
        episode_rewards=np.zeros(n_episodes))
    
    epsilon = e_initial
    for episode in tqdm(range(n_episodes)):

        #Make/update the policy
        policy = EgreedyPolicy(Q, epsilon, nA)

        # init total rewards per episode for statistics
        total_r = 0

        # Reset environment, Get observation s
        s = env.reset()

        # Decaying epsilon
        if epsilon > e_final:
            epsilon *= e_decay

        for t in itertools.count(): # Counts timesteps t for each episode
            
            # Get action given state s, choose action egreedily
            a = policy.step(s)

            # Perform action, observe R, S'
            s_prime, r, done, _ = env.step(a)

            # Since Frozenlake-v0 env returns r > 0 only when agent gets to goal
            if done:
                rr = 1 if r > 0 else -1
            else:
                rr = 0


            # Q-Update
            Q[s][a] = Q[s][a] + alpha * (rr + (gamma * np.max(Q[s_prime])) - Q[s][a])

            # state update
            s = s_prime

            total_r += r
            if done: 
                break

        stats.episode_lengths[episode] = t
        stats.episode_rewards[episode] = total_r
            
    return Q, stats

def sarsa(env, n_episodes, e_initial, e_final, e_decay, alpha=0.1, gamma=1.0):
    '''
    Implementation of the Q-Learning algorithm: Off-policy TD control.
    Follows an epsilon-greedy policy with parameters set by arguments.

    Args:
        env: OpenAI gym environment
        n_episodes: number of training eps
        alpha: learning rate for Q update
        gamma: discount factor for future reward estimates

    Returns:
        (Q, stats): Tuple of new state-action dictionary and training stats
    '''

    # number of Actions in env
    nA = env.action_space.n

    # initialize Q(S,A) : state-action -> action-value function
    Q = defaultdict(lambda: np.zeros(nA))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(n_episodes),
        episode_rewards=np.zeros(n_episodes))
    
    epsilon = e_initial
    for episode in tqdm(range(n_episodes)):

        #Make/update the policy
        policy = EgreedyPolicy(Q, epsilon, nA)

        # init total rewards per episode for statistics
        total_r = 0

        # Initilize environment, Get state [s]
        s = env.reset()

        # SARSA-only: Pick initial action
        a = policy.step(s)

        # Decaying epsilon
        if epsilon > e_final:
            epsilon *= e_decay

        for t in itertools.count(): # Counts timesteps t for each episode

            # Perform action, observe reward [r], next state [s']
            s_prime, r, done, _ = env.step(a)

            # SARSA: Pick the next action [a']
            a_prime = policy.step(s_prime)

            # Q-Update
            Q[s][a] = Q[s][a] + alpha * (r + (gamma * Q[s_prime][a_prime]) - Q[s][a])

            # state update
            s = s_prime

            # SARSA: action update
            a = a_prime

            total_r += r
            if done: 
                break

        stats.episode_lengths[episode] = t
        stats.episode_rewards[episode] = total_r
            
    return Q, stats

def sarsa_frozenlake(env, n_episodes, e_initial, e_final, e_decay, alpha=0.1, gamma=1.0):
    '''
    Implementation of the Q-Learning algorithm: Off-policy TD control.
    Follows an epsilon-greedy policy with parameters set by arguments.

    Args:
        env: OpenAI gym environment
        n_episodes: number of training eps
        alpha: learning rate for Q update
        gamma: discount factor for future reward estimates

    Returns:
        (Q, stats): Tuple of new state-action dictionary and training stats
    '''

    # number of Actions in env
    nA = env.action_space.n

    # initialize Q(S,A) : state-action -> action-value function
    Q = defaultdict(lambda: np.zeros(nA))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(n_episodes),
        episode_rewards=np.zeros(n_episodes))
    
    epsilon = e_initial
    for episode in tqdm(range(n_episodes)):

        #Make/update the policy
        policy = EgreedyPolicy(Q, epsilon, nA)

        # init total rewards per episode for statistics
        total_r = 0

        # Initilize environment, Get state [s]
        s = env.reset()

        # SARSA-only: Pick initial action
        a = policy.step(s)

        # Decaying epsilon
        if epsilon > e_final:
            epsilon *= e_decay

        for t in itertools.count(): # Counts timesteps t for each episode

            # Perform action, observe reward [r], next state [s']
            s_prime, r, done, _ = env.step(a)

            # SARSA: Pick the next action [a']
            a_prime = policy.step(s_prime)

            # Since Frozenlake-v0 env returns r > 0 only when agent gets to goal
            if done:
                rr = 1 if r > 0 else -1
            else:
                rr = 0

            # Q-Update
            Q[s][a] = Q[s][a] + alpha * (rr + (gamma * Q[s_prime][a_prime]) - Q[s][a])

            # state update
            s = s_prime
            
            # SARSA: action update
            a = a_prime

            total_r += r
            if done: 
                break

        stats.episode_lengths[episode] = t
        stats.episode_rewards[episode] = total_r
            
    return Q, stats

class OneHotEncoding(gym.ObservationWrapper):

    def __init__(self, env):
        super(OneHotEncoding, self).__init__(env)
        n_states = env.observation_space.n
        self.observation_space = gym.spaces.Box(0.0, 1.0, (n_states, ), dtype=np.float32)

    def observation(self, state):
        reward = np.copy(self.observation_space.low)
        reward[state] = 1.0
        return reward

class PolicyNN(nn.Module):

    def __init__(self, n_states, n_actions):
        super(PolicyNN, self).__init__()

        self.hidden = nn.Linear(n_states, 128)
        self.output = nn.Linear(128, n_actions)

    def forward(self, x):

        x = F.relu(self.hidden(x))
        x = self.output(x)

        return x
    
    def predict(self, x):

        x = self.forward(x)
        
        return F.softmax(x, dim=0)

def policy(net, state):

    with torch.no_grad():

        inputs = torch.tensor(state).cuda()
        preds = net.predict(inputs)
        preds = preds.cpu().numpy()

    # preds is the NN's predicted probability of taking an action.
    # The policy is to follow this distribution choosing actions.
    action = np.random.choice(len(preds), p=preds)

    return action

def cem_frozenlake(env, lr=1e-3, elite_ratio=0.3, n_episodes=100, n_epochs=1):

    net = PolicyNN(n_states=4*4, n_actions=4)
    net.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=lr)

    batch_size = int(elite_ratio * n_episodes)

    stats = plotting.EpisodeStats(
    episode_lengths=np.zeros(n_epochs*n_episodes),
    episode_rewards=np.zeros(n_epochs*n_episodes))

    for epoch in tqdm(range(n_epochs)):

        # Dataset
        all_states = []
        all_actions = []
        mean_ep_reward = []
        all_rewards = []

        for episode in range(n_episodes):

            # Store states and actions of this episode in:
            ep_states = []
            ep_actions = []
            ep_reward = 0
            total_r = 0.
            
            # Reset, get initial state
            s = env.reset()

            for t in itertools.count():

                a = policy(net, s)
                s_prime, r, done, _ = env.step(a)

                # Frozenlake: Save state-action-rewards for CEM
                ep_states.append(s)
                ep_actions.append(a)
                if done and r > 0:
                    ep_reward = 0.9**(len(ep_states))
                    # 6 actions to perfectly solve FrozenLake

                total_r += r
                if done:
                    break

                s = s_prime

            # Update stats
            stats.episode_lengths[epoch*n_episodes + episode] = t
            stats.episode_rewards[epoch*n_episodes + episode] = total_r
            
            # Append this episode to dataset
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
            all_rewards.extend([ep_reward] * (len(ep_states))) # each state is assigned the episode's reward.
            mean_ep_reward.append(ep_reward)

        #print('Epoch {} mean reward: {:.5f}'.format(
        #    epoch, np.mean(np.array(mean_ep_reward))))

        all_states, all_actions, all_rewards = map(
            np.array, [all_states, all_actions, all_rewards])

        best_episodes = np.argsort(-all_rewards)[:batch_size]

        x_states = all_states[best_episodes]
        y_actions = all_actions[best_episodes]

        # Training the neural network
        x_states, y_actions = map(torch.from_numpy, [x_states, y_actions])
        x_states, y_actions = map(lambda x: x.cuda(), [x_states, y_actions])

        preds = net(x_states)
        loss = loss_fn(preds, y_actions)
        loss.backward()
        optimizer.step()

    return stats

def cem_cliffwalking(env, lr=1e-3, elite_ratio=0.3, n_episodes=100, n_epochs=1):

    net = PolicyNN(n_states=4*12, n_actions=4)
    net.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=lr)

    batch_size = int(elite_ratio * n_episodes)

    stats = plotting.EpisodeStats(
    episode_lengths=np.zeros(n_epochs*n_episodes),
    episode_rewards=np.zeros(n_epochs*n_episodes))

    for epoch in tqdm(range(n_epochs)):

        # Dataset
        all_states = []
        all_actions = []
        mean_ep_reward = []
        all_rewards = []

        for episode in range(n_episodes):

            # Store states and actions of this episode in:
            ep_states = []
            ep_actions = []
            ep_reward = 0
            total_r = 0.
            
            # Reset, get initial state
            env.reset()
            s = env.get_state_onehot()

            for t in itertools.count():

                a = policy(net, s)
                s_prime, r, done, _ = env.step(a)
                s_prime = env.get_state_onehot()

                # Frozenlake: Save state-action-rewards for CEM
                ep_states.append(s)
                ep_actions.append(a)
                if done and r == -1:
                    ep_reward = 0.9**(len(ep_states))

                total_r += r
                if done:
                    break

                s = s_prime

            # Update stats
            stats.episode_lengths[epoch*n_episodes + episode] = t
            stats.episode_rewards[epoch*n_episodes + episode] = total_r
            
            # Append this episode to dataset
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
            all_rewards.extend([ep_reward] * (len(ep_states))) # each state is assigned the episode's reward.
            mean_ep_reward.append(ep_reward)

        #print('Epoch {} mean reward: {:.5f}'.format(
        #    epoch, np.mean(np.array(mean_ep_reward))))

        all_states, all_actions, all_rewards = map(
            np.array, [all_states, all_actions, all_rewards])

        best_episodes = np.argsort(-all_rewards)[:batch_size]

        x_states = all_states[best_episodes]
        y_actions = all_actions[best_episodes]

        # Training the neural network
        x_states, y_actions = map(torch.from_numpy, [x_states, y_actions])
        x_states, y_actions = map(lambda x: x.cuda(), [x_states, y_actions])

        preds = net(x_states)
        loss = loss_fn(preds, y_actions)
        loss.backward()
        optimizer.step()

    return stats

