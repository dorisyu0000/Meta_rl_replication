import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from scipy.special import softmax
from scipy.optimize import minimize

import gymnasium as gym

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from environment import *
from modules import *
from trainer import *
from models import *

np.random.seed(1)  # Fixing the random seed

num_nets = 100

for id in range(num_nets):

    net = torch.load('code_bandit2/nets/net' + str(id) + '.pth')

    env = DriftingBandit()
    env = MetaLearningWrapper(env)

    data = {
        'mus': [],
        'stages': [],
        'actions': [],
        'rewards': [],
    }

    num_simulation = 10

    for i in range(num_simulation):

        stage_seq = []
        action_seq = []
        reward_seq = []

        obs, info = env.reset()
        obs = torch.Tensor(obs).unsqueeze(dim = 0)

        done = False
        states_actor, states_critic = None, None

        with torch.no_grad():
            # iterate through a trial
            while not done:

                info_cache = info.copy()

                # step the net
                action, policy, log_prob, entropy, value, states_actor, states_critic = net(obs, states_actor, states_critic)
                action = torch.argmax(policy)

                # step the env
                obs, reward, done, _, info = env.step(action.item())
                obs = torch.Tensor(obs).unsqueeze(dim = 0)

                if info_cache['stage'] == 'decision':
                    # print(
                    #     'trial count:', info_cache['num_completed'], '|',
                    #     'stage:', info_cache['stage'], '|',
                    #     'correct choice:', np.argmax(info_cache['mus_seq'][info_cache['num_completed'], :]), '|',
                    #     'action:', action.item(), '|',
                    #     'reward:', round(reward, 3), '|',
                    #     'done:', done, '|',
                    # )
                    stage_seq.append(info_cache['stage'])
                    action_seq.append(action.item())
                    reward_seq.append(reward)
                
            # process the last timestep
            action, policy, log_prob, entropy, value, states_actor, states_critic = net(obs, states_actor, states_critic)
            action = torch.argmax(policy)


            data['mus'].append(env.mus_seq)
            data['stages'].append(stage_seq)
            data['actions'].append(np.array(action_seq))
            data['rewards'].append(np.array(reward_seq))



    param_init = [0.5, 10., 1.]

    tdagent = TDLearningAgent()
    spagent = SamplingAgent()

    best_params_td = tdagent.fit(data, param_init)
    best_params_sp = spagent.fit(data, param_init)
    
    pickle.dump(best_params_td, open('code_bandit2/params/params_td' + str(id) + '.p', 'wb'))
    pickle.dump(best_params_sp, open('code_bandit2/params/params_sp' + str(id) + '.p', 'wb'))

    print('td')
    print(best_params_td)
    print('sp')
    print(best_params_sp)
    print()