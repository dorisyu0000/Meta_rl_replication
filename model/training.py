import numpy as np
import time
import pickle

import gymnasium as gym

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from environment import *
from modules import *
from model.A2C import *

env = DriftingBandit()
env = MetaLearningWrapper(env)

net = RecurrentActorCriticPolicy(
    feature_dim = env.observation_space.shape[0],
    action_dim = env.action_space.n,
    policy_hidden_dim = 32,
    value_hidden_dim = 32,
    lstm_hidden_dim = 128,
)

a2c = A2C(
    net = net,
    env = env,
    lr = 3e-4,
    gamma = 0.2,
    beta_v = 0.5,
    beta_e = 1.,
    # lr_schedule = np.linspace(3e-4, 1e-4, num = 30000),
    entropy_schedule = np.linspace(1., 0.05, num = 80000)
)

losses, rewards = a2c.learn(num_episodes = 80000)
a2c.save_net('code_bandit2/net.pth')

data = {'losses': losses, 'rewards': rewards}
pickle.dump(data, open('code_bandit2/data.p', 'wb'))

plt.figure()
plt.plot(np.array(rewards).reshape(500, -1).mean(axis = 1))
plt.show()