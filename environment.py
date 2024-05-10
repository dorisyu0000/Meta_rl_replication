import numpy as np
import time
import os

import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from sb3_contrib import RecurrentPPO


class DriftingBandit(gym.Env):
    """
    A bandit environment.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self):
        """
        Construct an environment.
        """

        # task parameters
        self.num_bandit = 4
        self.num_trials = 150

        # bandit parameters
        self.lb, self.hb = 0, 100
        self.lamda = 0.9836 # decay parameter
        self.theta = 50. # decay center
        self.theta_d = 2.8 # sd for diffusion noise v
        self.theta_o = 4. # observation noise

        self.stage_dict = {
            'fixation': 1.,
            'decision': 0.,
        }

        # action and observation space
        self.action_space = Discrete(self.num_bandit + 1)
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = (1,))


    def reset(self, seed = None, option = {}):
        """
        Reset the environment.
        """

        self.num_completed = 0
        self.init_mus_seq()
        self.stage = 'fixation'

        obs = np.array(self.stage_dict[self.stage])

        info = {
            'stage': self.stage,
            'num_completed': self.num_completed,
            'mus_seq': self.mus_seq
        }

        return obs, info
    

    def step(self, action):
        """
        Step the environment.
        """

        done = False

        if self.stage == 'fixation':
            self.stage = 'decision'

            if action == self.num_bandit: # fixation action
                reward = 0.
            else:
                reward = -1.
        
        elif self.stage == 'decision':
            self.stage = 'fixation'

            if action == self.num_bandit: # fixation action
                reward = -1.
            else:
                reward = self.generate_reward(pointer = self.num_completed, choice = action)

            self.num_completed += 1

        obs = np.array(self.stage_dict[self.stage])
        
        if self.num_completed >= self.num_trials:
            done = True
        
        info = {
            'stage': self.stage,
            'num_completed': self.num_completed,
            'mus_seq': self.mus_seq
        }

        return obs, reward, done, False, info
    

    def init_mus_seq(self):
        """
        Generate a mean sequence.
        """

        self.mus_seq = []

        mus_t = np.random.uniform(low = self.lb, high = self.hb, size = self.num_bandit)
        self.mus_seq.append(mus_t)

        for _ in range(self.num_trials - 1):
            v = np.random.normal(loc = 0, scale = self.theta_d, size = self.num_bandit)
            mus_t = self.lamda * mus_t + (1 - self.lamda) * self.theta + v
            self.mus_seq.append(mus_t)
        
        self.mus_seq = np.array(self.mus_seq) # (num_trials, num_bandit)
        

    def generate_reward(self, pointer, choice):
        """
        Generate reward observation.
        """

        mu = self.mus_seq[pointer, choice]
        reward = np.random.normal(loc = mu, scale = self.theta_o, size = 1)[0]
        reward /= 50.

        return reward

        
    def one_hot_coding(self, num_classes, labels = None):
        """
        One-hot code nodes.
        """

        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]

        return labels_one_hot



class MetaLearningWrapper(Wrapper):
    """
    A meta-RL wrapper.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, env):
        """
        Construct an wrapper.
        """

        super().__init__(env)

        self.env = env
        self.one_hot_coding = env.get_wrapper_attr('one_hot_coding')

        self.init_prev_variables()

        new_observation_shape = (
            self.env.observation_space.shape[0] +
            self.env.action_space.n + # previous action
            1, # previous reward
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = new_observation_shape)


    def step(self, action):
        """
        Step the environment.
        """

        obs, reward, done, truncated, info = self.env.step(action)

        obs_wrapped = self.wrap_obs(obs)

        self.prev_action = action
        self.prev_reward = reward

        return obs_wrapped, reward, done, truncated, info
    

    def reset(self, seed = None, options = {}):
        """
        Reset the environment.
        """

        obs, info = self.env.reset()

        self.init_prev_variables()

        obs_wrapped = self.wrap_obs(obs)

        return obs_wrapped, info
    

    def init_prev_variables(self):
        """
        Reset previous variables.
        """

        self.prev_action = None
        self.prev_reward = 0.


    def wrap_obs(self, obs):
        """
        Wrap observation with previous variables.
        """

        obs_wrapped = np.hstack([
            obs,
            self.one_hot_coding(num_classes = self.env.action_space.n, labels = self.prev_action),
            self.prev_reward
        ])
        return obs_wrapped



if __name__ == '__main__':
    
    env = DriftingBandit()
    env = MetaLearningWrapper(env)
    

    # model = RecurrentPPO(
    #     policy = 'MlpLstmPolicy',
    #     env = env,
    #     verbose = 1,
    #     learning_rate = 1e-4,
    #     n_steps = 20,
    #     gamma = 0.9,
    #     ent_coef = 0.05,
    # )

    # model.learn(total_timesteps = 1000000)

    for i in range(50):

        obs, _ = env.reset()
        print('initial obs:', obs)
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            print(
                'obs:', obs, '|',
                'action', action, '|',
                'reward:', reward, '|',
                'done:', done
            )
