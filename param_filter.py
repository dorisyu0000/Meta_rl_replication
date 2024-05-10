import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

import gymnasium as gym

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from environment import *
from modules import *
from trainer import *
from models import *

path = 'code_bandit2/params'

num_net = 100
num_fitting = 10

best_indices_td = np.zeros((num_net,), dtype = int)
best_indices_sp = np.zeros((num_net,), dtype = int)

best_params_td = np.zeros((num_net, 3))
best_params_sp = np.zeros((num_net, 3))

best_nll_td = np.zeros((num_net,))
best_nll_sp = np.zeros((num_net,))


for i in range(num_net):

    params_td_net = np.zeros((num_fitting, 3))
    params_sp_net = np.zeros((num_fitting, 3))

    nll_td_net = np.zeros((num_fitting,))
    nll_sp_net = np.zeros((num_fitting,))

    for j in range(num_fitting):

        params_td = pickle.load(open('code_bandit2/params/params_td' + str(i) + '_' + str(j) + '.p', 'rb'))
        params_sp = pickle.load(open('code_bandit2/params/params_sp' + str(i) + '_' + str(j) + '.p', 'rb'))

        params_td_net[j, :] = params_td['x']
        params_sp_net[j, :] = params_sp['x']

        nll_td_net[j] = params_td['fun']
        nll_sp_net[j] = params_sp['fun']

    best_indices_td[i] = np.argmin(nll_td_net)
    best_indices_sp[i] = np.argmin(nll_sp_net)

    best_params_td[i, :] = params_td_net[np.argmin(nll_td_net), np.arange(3)]
    best_params_sp[i, :] = params_sp_net[np.argmin(nll_sp_net), np.arange(3)]

    best_nll_td[i] = np.mean(nll_td_net)
    best_nll_sp[i] = np.mean(nll_sp_net)

data = {
    'best_indices_td': best_indices_td,
    'best_indices_sp': best_indices_sp,
    'best_params_td': best_params_td,
    'best_params_sp': best_params_sp,
    'best_nll_td': best_nll_td,
    'best_nll_sp': best_nll_sp,

}

pickle.dump(data, open('code_bandit2/data.p', 'wb'))