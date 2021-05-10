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