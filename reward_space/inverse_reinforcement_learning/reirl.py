import numpy.linalg as la
from reward_space.inverse_reinforcement_learning.utils import *


class RelativeEntropyIRL(object):


    eps = 1e-24

    def __init__(self,
                 reward_features,
                 trajectories_expert,
                 trajectories_random,
                 gamma,
                 horizon,
                 n_states,
                 n_actions,
                 learning_rate=0.01,
                 max_iter=100,
                 type_='state',
                 gradient_method='linear',
                 evaluation_horizon=100):

        # transition model: tensor (n_states, n_actions, n_states)

        self.reward_features = reward_features
        self.trajectories_expert = trajectories_expert
        self.trajectories_random = trajectories_random
        self.gamma = gamma
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        if not type_ in ['state', 'state-action']:
            raise ValueError()
        self.type_ = type_

        if not gradient_method in ['linear', 'exponentiated']:
            raise ValueError()
        self.gradient_method = gradient_method

        self.evaluation_horizon = evaluation_horizon

        self.n_states, self.n_actions = n_states, n_actions
        self.n_features = reward_features.shape[1]


    def fit(self, verbose=False):

        #Compute features expectations
        expert_feature_expectations = compute_feature_expectations(self.reward_features,
                                                            self.trajectories_expert,
                                                            np.arange(self.n_states),
                                                            np.arange(self.n_actions),
                                                            self.gamma,
                                                            self.horizon,
                                                            self.type_)

        random_feature_expectations = compute_feature_expectations(self.reward_features,
                                                            self.trajectories_random,
                                                            np.arange(self.n_states),
                                                            np.arange(self.n_actions),
                                                            self.gamma,
                                                            self.horizon,
                                                            self.type_,
                                                            'separated')

        n_random_trajectories = int(sum(self.trajectories_random[:, -1]))
        importance_sampling = np.zeros(n_random_trajectories)

        #Weights initialization
        w = np.ones(self.n_features) / self.n_features

        #Gradient descent
        for i in range(self.max_iter):

            if verbose:
                print('Iteration %s/%s' % (i + 1, self.max_iter))

            for j in range(n_random_trajectories):
                importance_sampling[j] = np.exp(np.dot(random_feature_expectations[j], w))
            importance_sampling /= np.sum(importance_sampling, axis=0)
            weighted_sum = np.sum(np.multiply(np.array([importance_sampling, ] * random_feature_expectations.shape[1]).T, random_feature_expectations), axis=0)


            w += self.learning_rate * (expert_feature_expectations - weighted_sum)

            # One weird trick to ensure that the weights don't blow up the objective.
            w = w / np.linalg.norm(w, keepdims=True)

        w /= w.sum()
        reward = np.dot(self.reward_features, w)
        return reward.ravel()