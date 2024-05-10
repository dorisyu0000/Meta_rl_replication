import numpy as np
import time
from scipy.special import softmax
from scipy.optimize import minimize

import matplotlib.pyplot as plt


class TDLearningAgent:
    """
    A TD learning agent.
    """
    
    def __init__(self):
        """
        Initialize the agent.
        """

        # task parameters
        self.num_bandit = 4
        self.num_trials = 150

        # initial Q value
        self.q_init = np.ones(self.num_bandit) * 50. / 50.


    def get_policies(self, actions, rewards, param):
        """
        Compute policies for an episode.
        """

        # get parameters
        alpha_td = param['alpha_td']
        beta_td = param['beta_td']
        beta_c = param['beta_c']

        # initialize Q values
        qs = np.zeros((self.num_trials, self.num_bandit))
        q = self.q_init.copy()
        qs[0, :] = q

        # initialize indicators
        indicators = np.zeros((self.num_trials, self.num_bandit))

        # iterately compute Q values
        for t in range(1, self.num_trials):
            a = actions[t - 1]
            r = rewards[t - 1]
            q[a] = q[a] + alpha_td * (r - q[a])

            # record results
            qs[t, :] = q
            indicators[t, a] = 1.
        
        # compute policies
        logits = beta_td * qs + beta_c * indicators # (num_trials, num_bandit)
        policies = softmax(logits, axis = 1) # (num_trials, num_bandit)

        return policies


    def get_nll(self, actions, rewards, param):
        """
        Compute negative log likelihoods for an episode.
        """

        # get policies
        policies = self.get_policies(actions, rewards, param) # (num_trials, num_bandit)

        # compute action probabilities
        probs = policies[np.arange(self.num_trials), actions]
        
        # compute negative log likelihoods
        nll = -np.log(probs).sum()

        return nll


    def loss_function(self, data, param):
        """
        Loss functions across episodes.
        """

        nll_total = 0.
        num_episode = len(data['actions'])

        # sum loss functions over episodes
        for i in range(num_episode):
            actions = data['actions'][i]
            rewards = data['rewards'][i]

            nll_episode = self.get_nll(actions, rewards, param)

            nll_total += nll_episode
        
        return nll_total


    def fit(self, data, param_init):
        """
        Fit the model.
        """

        # Constrain parameter ranges
        bounds = [(0., 1.), (0., None), (0., None)]

        # define loss function
        loss = lambda param: self.loss_function(
            data = data,
            param = {
                'alpha_td': param[0],
                'beta_td': param[1],
                'beta_c': param[2]
            }
        )

        # perform MLE
        result = minimize(loss, param_init, method = 'Nelder-Mead', bounds = bounds)

        return result
    

    def simulate(self, env, param):
        """
        Simulate the model.
        """
        # get parameters
        alpha_td = param[0]
        beta_td = param[1]
        beta_c = param[2]

        # initialize rewards and actions
        actions = np.zeros((self.num_trials,), dtype = int)
        rewards = np.zeros((self.num_trials,))

        # initialize Q values
        qs = np.zeros((self.num_trials, self.num_bandit))
        q = self.q_init.copy()
        qs[0, :] = q

        # iterately make decisions
        for t in range(0, self.num_trials):
            if t == 0:
                indicator = np.zeros((4,))
            else:
                indicator = np.eye(self.num_bandit)[actions[t - 1]]
                    
            logits = beta_td * qs[t, :] + beta_c * indicator
            policy = softmax(logits)

            # action = np.random.choice(self.num_bandit, p = policy)
            action = np.argmax(policy)

            mu = env.mus_seq[t, action]
            reward = np.random.normal(loc = mu, scale = env.theta_o, size = 1) / 50.

            actions[t] = action
            rewards[t] = reward

            if t < self.num_trials - 1:
                qs[t + 1] = qs[t]
                qs[t + 1, action] = qs[t, action] + alpha_td * (reward - qs[t, action])

        
        return actions, rewards




    


class SamplingAgent:
    """
    A TD learning agent.
    """
    
    def __init__(self):
        """
        Initialize the agent.
        """

        # task parameters
        self.num_bandit = 4
        self.num_trials = 150

        # initial reward sample value
        self.r_init = 50. / 50.
    

    def get_policies(self, actions, rewards, param):
        """
        Compute negative log likelihoods for an episode.
        """

        # initialize reward histories (inserting r0)
        rs_histories = [np.array([self.r_init]) for _ in range(self.num_bandit)]

        # initialize policies
        policies = np.zeros((self.num_trials, self.num_bandit))

        for t in range(self.num_trials):
            # compute history sizes for each arm
            history_sizes = [len(_) for _ in rs_histories]

            # compute sample probabilities in the history
            sample_probs = self.get_sample_probs(history_sizes, param)

            # get indicator
            if t == 0:
                indicator = np.zeros((4,))
            else:
                indicator = np.eye(self.num_bandit)[actions[t - 1]]
            
            # compute policy
            policy = self.get_policy(rs_histories, sample_probs, indicator, param)
            policies[t, :] = policy

            # append reward history
            a = actions[t]
            r = rewards[t]
            rs_histories[a] = np.append(rs_histories[a], r)
        
        return policies
    

    def get_nll(self, actions, rewards, param):
        """
        Compute negative log likelihoods for an episode.
        """

        # get policies
        policies = self.get_policies(actions, rewards, param) # (num_trials, num_bandit)

        # compute action probabilities
        probs = policies[np.arange(self.num_trials), actions]
        
        # compute negative log likelihoods
        nll = -np.log(probs).sum()

        return nll
        

    def get_sample_probs(self, history_sizes, param):
        """
        Compute sample probabilities based on reward history sizes.
        """

        alpha_sample = param['alpha_sample']

        # initialize sample probabilities
        sample_probs = []

        # loop over arms
        for arm in range(self.num_bandit):
            # remove r0 when computing history sizes
            history_size_arm = history_sizes[arm] - 1 # remove r0
            
            # compute probability sequence for history
            sample_probs_arm = alpha_sample * (1 - alpha_sample) ** (history_size_arm - np.arange(1, history_size_arm + 1, 1)) # ? this can change to -1
            # the sample_probs_arm increases with index

            # insert p0
            sample_probs_arm = np.insert(sample_probs_arm, 0, 1 - sample_probs_arm.sum())

            sample_probs.append(sample_probs_arm)
        
        return sample_probs
    

    def get_policy(self, rs_histories, sample_probs, indicator, param):
        """
        Compute action probabilities based on reward histories and sample probabilities.
        """

        # get temperatures
        beta_sample = param['beta_sample']
        beta_c = param['beta_c']

        # get combinations of all histories
        rs_grid = np.meshgrid(rs_histories[0], rs_histories[1], rs_histories[2], rs_histories[3])
        rs_combinations = np.array([g.ravel() for g in rs_grid]).T # (n_combinations, n_bandit)

        # get corresponding combinations of all sample probabilities
        ps_grid = np.meshgrid(sample_probs[0], sample_probs[1], sample_probs[2], sample_probs[3])
        ps_combinations = np.array([g.ravel() for g in ps_grid]).T # (n_combinations, n_bandit)

        # compute policy
        exp_combinations = np.exp(beta_sample * rs_combinations + beta_c * indicator)# (n_combinations, n_bandit)
        logits_combinations = exp_combinations / exp_combinations.sum(axis = 1)[:, None] # (n_combinations, n_bandit)
        policy = (ps_combinations.prod(axis = 1)[:, None] * logits_combinations).sum(axis = 0) # (n_bandit.)

        return policy


    def loss_function(self, data, param):
        """
        Loss functions across episodes.
        """

        nll_total = 0.
        num_episode = len(data['actions'])

        # sum loss functions over episodes
        for i in range(num_episode):
            actions = data['actions'][i]
            rewards = data['rewards'][i]

            nll_episode = self.get_nll(actions, rewards, param)

            nll_total += nll_episode
        
        return nll_total


    def fit(self, data, param_init):
        """
        Fit the model.
        """

        # Constrain parameter ranges
        bounds = [(0., 1.), (0., None), (0., None)]

        # define loss function
        loss = lambda param: self.loss_function(
            data = data,
            param = {
                'alpha_sample': param[0],
                'beta_sample': param[1],
                'beta_c': param[2]
            }
        )

        # perform MLE
        result = minimize(loss, param_init, method = 'Nelder-Mead', bounds = bounds)

        return result
    

    def simulate(self, env, param):
        """
        Simulate the model.
        """
        # get parameters
        alpha_sample = param[0]
        beta_sample = param[1]
        beta_c = param[2]


        # initialize rewards and actions
        actions = np.zeros((self.num_trials,), dtype = int)
        rewards = np.zeros((self.num_trials,))

        # initialize reward histories (inserting r0)
        rs_histories = [np.array([self.r_init]) for _ in range(self.num_bandit)]

        # iterately make decisions
        for t in range(self.num_trials):
            # compute history sizes for each arm
            history_sizes = [len(_) for _ in rs_histories]

            # compute sample probabilities in the history
            sample_probs = self.get_sample_probs(history_sizes, {'alpha_sample': alpha_sample})

            # get indicator
            if t == 0:
                indicator = np.zeros((4,))
            else:
                indicator = np.eye(self.num_bandit)[actions[t - 1]]
            
            # compute policy
            policy = self.get_policy(rs_histories, sample_probs, indicator, {'beta_sample': beta_sample, 'beta_c': beta_c})
            
            action = np.argmax(policy)

            mu = env.mus_seq[t, action]
            reward = np.random.normal(loc = mu, scale = env.theta_o, size = 1) / 50.

            # append reward history
            actions[t] = action
            rewards[t] = reward
            rs_histories[action] = np.append(rs_histories[action], reward)
        
        return actions, rewards


if __name__ == '__main__':

    np.random.seed(42)  # Fixing the random seed

    # actions = np.random.randint(low = 0, high = 4, size = 100)
    # rewards = np.random.rand(100) * 100
    # param = {
    #     'alpha_sample': 0.1,
    #     'beta_sample': 0.1,
    #     'beta_c': 0.1,
    # }

    # spagent = SamplingAgent()


    # tic = time.time()

    # nll = spagent.get_nll(actions, rewards, param)


    # toc = time.time()
    # print(toc - tic)





    # tic = time.time()

    # spagent = SamplingAgent()

    # data = {
    #     'actions': [],
    #     'rewards': [],
    # }
    # for i in range(100):
    #     data['actions'].append(np.random.randint(low = 0, high = 4, size = 150))
    #     data['rewards'].append(np.random.rand(150) * 2)

    # param_init = [0.1, 0.1, 0.1]

    # best_params = spagent.fit(data, param_init)
    # print("Best Parameters:", best_params)

    # toc = time.time()
    # print(toc - tic)

