import numpy as np
from gym.utils import seeding
from taxi_policy_iteration import compute_policy
from scipy.stats import multivariate_normal
import numpy.linalg as la

class Policy(object):
    
    '''
    Abstract class
    '''

    def draw_action(self, state, done):
        pass


class SimplePolicy(Policy):
    
    '''
    Deterministic policy with parameter K
    '''
    
    def __init__(self, K, action_bounds=None):
        self.K = np.array(K, ndmin=2)
        self.n_dim = self.K.shape[0]
        if action_bounds is None:
            self.action_bounds = np.array([[-np.inf] *self.n_dim, [np.inf] *self.n_dim], ndmin=2)
        else:
            self.action_bounds = np.array(action_bounds)
    
    def draw_action(self, state, done):
        action = np.dot(self.K, state)
        bound_action = self.check_action_bounds(action)
        return bound_action
    
    def check_action_bounds(self, action):
        return np.clip(action, self.action_bounds[0], self.action_bounds[1])

        
class GaussianPolicy1D(SimplePolicy):
    
    '''
    Gaussian policy with parameter K for the mean and fixed variance
    just for 1 dimension lqr
    '''

    def __init__(self, K, sigma, action_bounds=None):
        SimplePolicy.__init__(self, K, action_bounds)
        self.sigma = sigma
        self.seed()

    def get_dim(self):
        return 1

    def set_parameter(self, K, build_gradient=True, build_hessian=True):
        self.K = K

    def draw_action(self, state, done):
        state = np.array(state, ndmin=1)
        action = np.dot(self.K, state) + self.np_random.randn() * self.sigma
        #bound_action = self.check_action_bounds(action)
        #return bound_action
        return action
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def pdf(self, state, action):
        return np.array(1. / np.sqrt(2 * np.pi * self.sigma ** 2) * np.exp(- 1. / 2 * (action - self.K * state) ** 2 / self.sigma ** 2), ndmin=1)


    def gradient_log(self, state, action, type_='state-action'):
        if type_ == 'state-action':
            return np.array(np.array(state/np.power(self.sigma,2)*(action - np.dot(self.K,state))).ravel(), ndmin=1)
        elif type_ == 'list':
            return map(lambda s,a: self.gradient_log(s, a), state, action)

    def hessian_log(self, state, action):
        return np.array(- state ** 2 / self.sigma ** 2, ndmin=2)
        
        
class GaussianPolicy(SimplePolicy):
    
    '''
    Gaussian policy with parameter K for the mean and fixed variance
    for any dimension
    TBR
    '''
    
    def __init__(self, K, covar):
        SimplePolicy.__init__(self, K)
        self.K = np.array(K, ndmin=2)
        self.ndim = self.K.shape[0]
        self.covar = np.array(covar, ndmin=2)
        self.seed()

    def draw_action(self, state, done):
        state = np.array(state, ndmin=1)
        mean = np.dot(self.K, state)
        action = self.np_random.multivariate_normal(mean, self.covar)
        return action
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def set_parameter(self, K, build_gradient=True, build_hessian=True):
        self.K = K

    def pdf(self, state, action):
        mean = np.dot(self.K, state[:, np.newaxis])
        return multivariate_normal.pdf(action, mean.ravel(), self.covar)

    def gradient_log(self, state, action, type_='state-action'):
        if type_ == 'state-action':
            deriv = np.transpose(np.tensordot(np.eye(self.ndim), state, axes=0).squeeze(),
                         (0, 2, 1))
            mean = np.dot(self.K, state)
            sol = la.solve(self.covar, mean)
            return np.array(np.tensordot(deriv, sol[:, np.newaxis], axes=1).squeeze(), ndmin=2)
        elif type_ == 'list':
            return map(lambda s,a: self.gradient_log(s, a), state, action)

    def hessian_log(self, state, action):
        deriv = np.transpose(
            np.tensordot(np.eye(self.ndim), state, axes=0).squeeze(),
            (0, 2, 1))
        mean = np.dot(self.K, state)
        sol = la.solve(self.covar, mean)
        return np.array(np.tensordot(np.tensordot(deriv, deriv, axes=1), sol[:, np.newaxis][:, np.newaxis], axes=1).squeeze(), ndmin=2)


class TaxiEnvPolicy(SimplePolicy):

    def __init__(self):
        self.policy = compute_policy()
        self.nS = 500
        self.nA = 6
        self.pi = np.zeros((self.nS, self.nA))
        self.pi2 = np.zeros((self.nS, self.nS * self.nA))
        s = np.array(self.policy.keys())
        a = np.array(self.policy.values())
        self.pi[s, a] = 1.
        self.pi2[s, s * self.nA + a] = 1.
        self.seed()

    def draw_action(self, state, done):
        action = self.np_random.choice(6, p=self.pi[np.asscalar(state)])
        return action

    def get_distribution(self):
        return self.pi2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

class TaxiEnvRandomPolicy(SimplePolicy):

    def __init__(self):
        self.nS = 500
        self.nA = 6
        self.PI = np.ones((self.nS, self.nA)) / self.nA
        self.PI2 = np.zeros((self.nS, self.nS * self.nA))
        self.PI2[np.repeat(np.arange(self.nS),self.nA), np.arange(self.nS*self.nA)] = 1. / self.nA
        self.seed()

    def draw_action(self, state, done):
        action = self.np_random.choice(6, p=self.PI[np.asscalar(state)])
        return action

    def get_distribution(self):
        return self.PI2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)


class BoltzmannPolicy(Policy):

    def __init__(self, features, parameters):
        self.features = features
        self.parameters = parameters
        self.n_states = self.features.shape[0]
        self.n_actions = self.parameters.shape[0]
        self.n_parameters = self.features.shape[1]

        self.state_action_features = np.zeros((self.n_states * self.n_actions, self.n_actions * self.n_parameters))
        self.state_action_parameters = parameters.ravel()[:, np.newaxis]
        row_index = np.repeat(np.arange(self.n_states * self.n_actions), self.n_parameters)
        col_index = np.tile(np.arange(self.n_parameters * self.n_actions), self.n_states)
        features_repeated = np.repeat(features, self.n_actions, axis=0).ravel()
        self.state_action_features[row_index, col_index] = features_repeated

        self._build_density()
        self._build_grad_hess()

        self.seed()

    def set_parameter(self, new_parameter, build_gradient=True, build_hessian=True):
        self.state_action_parameters = np.copy(new_parameter)
        self.parameters = self.state_action_parameters.reshape((self.n_actions, self.n_parameters))

        self._build_density()
        if build_gradient or build_hessian:
            self._build_grad_hess(build_hessian)

        self.seed()

    def _build_density(self):
        numerators = np.exp(np.dot(self.features, self.parameters.T))
        denominators = np.sum(numerators, axis=1)[:, np.newaxis]

        self.pi = numerators / denominators
        self.pi2 = np.zeros((self.n_states, self.n_actions * self.n_states))
        row_index = np.arange(self.n_states)[:, np.newaxis]
        col_index = np.arange(self.n_states * self.n_actions).reshape(self.n_states, self.n_actions)
        self.pi2[row_index, col_index] = self.pi

    def _build_grad_hess(self, build_hessian=True):
        self.grad_log = np.zeros((self.n_states * self.n_actions,
                                  self.n_parameters * self.n_actions))
        self.hess_log = np.zeros((self.n_states * self.n_actions,
                                  self.n_parameters * self.n_actions,
                                  self.n_parameters * self.n_actions))
        #Compute the gradient for all (s,a) pairs
        for state in range(self.n_states):

            num = den = 0
            for action in range(self.n_actions):
                index = state * self.n_actions + action
                exponential = np.exp(np.dot(self.state_action_features[index],\
                                            self.state_action_parameters))
                num += self.state_action_features[index] * exponential
                den += exponential

            for action in range(self.n_actions):
                index = state * self.n_actions + action
                self.grad_log[index] = self.state_action_features[index] - num / den

        #Compute the hessian for all (s,a) pairs
        if build_hessian:
            for state in range(self.n_states):

                num1 = num2 = den1 = 0
                for action in range(self.n_actions):
                    index = state * self.n_actions + action
                    exponential = np.exp(np.dot(self.state_action_features[index], \
                                                self.state_action_parameters))
                    num1 += np.outer(self.state_action_features[index],\
                                     self.state_action_features[index]) * exponential
                    num2 += self.state_action_features[index] * exponential
                    den1 += exponential

                num = num1 * den1 - np.outer(num2, num2)
                den = den1 ** 2

                for action in range(self.n_actions):
                    index = state * self.n_actions + action
                    self.hess_log[index] = - num / den

    def pdf(self, state, action):
        num = np.exp(np.dot(self.features[state], self.parameters[action].T))
        den = np.sum(np.exp(np.dot(self.features[state], self.parameters.T)))
        return num / den

    def get_pi(self, type_='state-action'):
        if type_ == 'state-action':
            return self.pi2
        elif type_ == 'state':
            return self.pi
        elif type_ == 'function':
            return self.pdf
        else:
            raise NotImplementedError

    def gradient_log(self, states=None, actions=None, type_='state-action'):
        if type_ == 'state-action':
            return self.grad_log
        elif type_ == 'list':
            return np.array(map(lambda s,a: self.grad_log[int(s) * self.n_actions + int(a)], states, actions))
        else:
            raise NotImplementedError

    def hessian_log(self, states=None, actions=None, type_='state-action'):
        if type_ == 'state-action':
            return self.hess_log
        elif type_ == 'list':
            return np.array(map(
                lambda s, a: self.hess_log[int(s) * self.n_actions + int(a)],
                states, actions))
        else:
            raise NotImplementedError

    def draw_action(self, state, done):
        action = self.np_random.choice(self.n_actions, p=self.pi[np.asscalar(state)])
        return action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def get_n_parameters(self):
        return self.get_dim()

    def get_dim(self):
        return self.state_action_parameters.shape[0]


class EpsilonGreedyBoltzmannPolicy(Policy):

    def __init__(self, epsilon, features, parameters):
        self.epsilon = epsilon
        self.features = features
        self.parameters = parameters
        self.n_states = self.features.shape[0]
        self.n_actions = self.parameters.shape[0]
        self.n_parameters = self.features.shape[1]

        self.state_action_features = np.zeros((self.n_states * self.n_actions, self.n_actions * self.n_parameters))
        self.state_action_parameters = parameters.ravel()[:, np.newaxis]
        row_index = np.repeat(np.arange(self.n_states * self.n_actions), self.n_parameters)
        col_index = np.tile(np.arange(self.n_parameters * self.n_actions), self.n_states)
        features_repeated = np.repeat(features, self.n_actions, axis=0).ravel()
        self.state_action_features[row_index, col_index] = features_repeated

        self._build_density()
        self._build_grad_hess()

        self.seed()

    def __str__(self):
        return str(self.__class__) + ' epsilon=' + str(self.epsilon)

    def set_parameter(self, new_parameter, build_gradient=True, build_hessian=True):
        self.state_action_parameters = np.copy(new_parameter)
        self.parameters = self.state_action_parameters.reshape((self.n_actions, self.n_parameters))

        self._build_density()
        if build_gradient or build_hessian:
            self._build_grad_hess(build_hessian)

        self.seed()

    def _build_density(self):
        numerators = np.exp(np.dot(self.features, self.parameters.T))
        denominators = np.sum(numerators, axis=1)[:, np.newaxis]

        self.pi = numerators / denominators * (1. - self.epsilon) + self.epsilon / self.n_actions
        self.pi2 = np.zeros((self.n_states, self.n_actions * self.n_states))
        row_index = np.arange(self.n_states)[:, np.newaxis]
        col_index = np.arange(self.n_states * self.n_actions).reshape(self.n_states, self.n_actions)
        self.pi2[row_index, col_index] = self.pi

    def _build_grad_hess(self, build_hessian=True):
        self.grad_log = np.zeros((self.n_states * self.n_actions,
                                  self.n_parameters * self.n_actions))
        self.hess_log = np.zeros((self.n_states * self.n_actions,
                                  self.n_parameters * self.n_actions,
                                  self.n_parameters * self.n_actions))
        #Compute the gradient and hessian for all (s,a) pairs
        for state in range(self.n_states):

            B_s = grad_B_s = hess_B_s = 0.
            for action in range(self.n_actions):
                index = state * self.n_actions + action
                feature = self.state_action_features[index]
                exponential = np.exp(np.dot(feature, self.state_action_parameters))
                if build_hessian:
                    hess_B_s += np.outer(feature, feature) * exponential
                grad_B_s += feature * exponential
                B_s += exponential

            grad_log_B_s = grad_B_s / B_s
            grad_hess_B_s = (hess_B_s * B_s - np.outer(grad_B_s, grad_B_s)) / B_s ** 2

            for action in range(self.n_actions):
                index = state * self.n_actions + action
                feature = self.state_action_features[index]
                exponential = np.exp(np.dot(feature, self.state_action_parameters))
                A_sa = exponential
                grad_A_sa = feature * exponential
                C_sa = (1. - self.epsilon) * A_sa + self.epsilon * B_s
                grad_C_sa = (1. - self.epsilon) * grad_A_sa + self.epsilon * grad_B_s
                self.grad_log[index] = grad_C_sa / C_sa - grad_log_B_s
                if build_hessian:
                    hess_A_sa = np.outer(feature, feature) * exponential
                    hess_C_sa = (1. - self.epsilon) * hess_A_sa + self.epsilon * hess_B_s
                    self.hess_log[index] = (hess_C_sa * C_sa - np.outer(grad_C_sa, \
                        grad_C_sa)) / C_sa ** 2 - grad_hess_B_s


    def pdf(self, state, action):
        num = np.exp(np.dot(self.features[state], self.parameters[action].T))
        den = np.sum(np.exp(np.dot(self.features[state], self.parameters.T)))
        return num / den

    def get_pi(self, type_='state-action'):
        if type_ == 'state-action':
            return self.pi2
        elif type_ == 'state':
            return self.pi
        elif type_ == 'function':
            return self.pdf
        else:
            raise NotImplementedError

    def gradient_log(self, states=None, actions=None, type_='state-action'):
        if type_ == 'state-action':
            return self.grad_log
        elif type_ == 'list':
            return np.array(map(lambda s,a: self.grad_log[int(s) * self.n_actions + int(a)], states, actions))
        else:
            raise NotImplementedError

    def hessian_log(self, type_='state-action'):
        if type_ == 'state-action':
            return self.hess_log
        else:
            raise NotImplementedError

    def draw_action(self, state, done):
        action = self.np_random.choice(self.n_actions, p=self.pi[np.asscalar(state)])
        return action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def get_dim(self):
        return self.state_action_parameters.shape[0]

class TabularPolicy(Policy):

    def __init__(self, probability_table):
        self.pi = probability_table

        self.n_states, self.n_actions = probability_table.shape
        self.n_state_actions = self.n_actions * self.n_states
        self.pi2 = np.zeros((self.n_states, self.n_state_actions))

        rows = np.repeat(np.arange(self.n_states), self.n_actions)
        cols = np.arange(self.n_state_actions)
        self.pi2[rows, cols] = self.pi.ravel()
        self.seed()

    def draw_action(self, state, done):
        action = self.np_random.choice(self.n_actions, p=self.pi[np.asscalar(state)])
        return action

    def get_distribution(self):
        return self.pi2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)


class RBFGaussianPolicy(SimplePolicy):

    def __init__(self,
                 centers,
                 parameters,
                 sigma,
                 radial_basis='gaussian',
                 radial_basis_parameters=None):

        self.centers = centers
        self.n_centers = self.centers.shape[0]
        self.parameters = parameters.ravel()[:, np.newaxis]
        self.sigma = sigma

        if radial_basis == 'gaussian':
            self.radial_basis = lambda x, center: np.exp(-radial_basis_parameters \
                                                         * la.norm(x - center))
        else:
            raise ValueError()

        self.seed()

    def get_dim(self):
        return self.parameters.shape[0]

    def set_parameter(self, parameter, build_gradient=True, build_hessian=True):
        self.parameters = parameter.ravel()[:, np.newaxis]

    def _compute_mean(self, state):
        rbf = [self.radial_basis(state, self.centers[i])
               for i in range(self.n_centers)]
        mean = np.dot(self.parameters.ravel(), rbf)
        return mean

    def draw_action(self, state, done):
        action = self._compute_mean(state) + self.np_random.randn() * self.sigma
        return action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def pdf(self, state, action):
        return np.array(1. / np.sqrt(2 * np.pi * self.sigma ** 2) * np.exp(
            - 1. / 2 * (action - self._compute_mean(state)) ** 2 / self.sigma ** 2),
                        ndmin=1)

    def gradient_log(self, state, action, type_='state-action'):

        if type_ == 'state-action':
            rbf = [self.radial_basis(state, self.centers[i])
                   for i in range(self.n_centers)]
            mean = np.dot(self.parameters.ravel(), rbf)
            gradient = (action - mean) / self.sigma ** 2 * np.array(rbf)
            return np.array(gradient.ravel(), ndmin=1)
        elif type_ == 'list':
            return map(lambda s, a: self.gradient_log(s, a), state, action)

    def hessian_log(self, state, action):
        rbf = [self.radial_basis(state, self.centers[i])
               for i in range(self.n_centers)]
        mean = np.dot(self.parameters.ravel(), rbf)
        hessian = (action - mean) / self.sigma ** 2 * np.outer(np.array(rbf), np.array(rbf))
        return np.array(hessian, ndmin=2)


class BivariateGaussianPolicy(GaussianPolicy):

    def __init__(self, k, sigma, rho):
        self.k = self.k = np.array(k).ravel()
        self.sigma = np.array(sigma)
        self.rho = rho
        K = np.diag(self.k)
        covar = np.diag(self.sigma ** 2)
        covar[0, 1] = covar[1, 0] = np.prod(self.sigma) * self.rho
        GaussianPolicy.__init__(self, K, covar)

    def gradient_log(self, state, action, type_='state-action'):
        if type_  == 'state-action':
            grad = 1. / (2 * (1 - self.rho ** 2)) * np.array([2 * state[i] / (self.sigma[i] ** 2) * \
                (action[i] - self.k[i] * state[i]) - 2 * self.rho / np.prod(self.sigma) * \
                state[i] * (action[1-i] - self.k[1-i]*state[1-i]) for i in range(2)])
            return np.array(grad)
        elif type_ == 'list':
            return map(lambda s, a: self.gradient_log(s, a), state, action)

    def hessian_log(self, state, action, type_='state-action'):
        if type_ == 'state-action':
            d = 1. / (2 * (1 - self.rho ** 2)) * np.array([-2 * state[i] ** 2/ (self.sigma[i] ** 2) for i in range(2)])
            cd = 1. / (2 * (1 - self.rho ** 2)) * np.array([ 2 * self.rho / np.prod(self.sigma) * \
                    state[i] *state[1-i] for i in range(2)])

            hess = np.diag(d)
            hess[0, 1], hess[1, 0] = cd
            return hess
        elif type_ == 'list':
            return map(lambda s, a: self.hessian_log(s, a), state, action)

    def get_dim(self):
        return 2

    def get_n_parameters(self):
        return self.get_dim()

    def set_parameter(self, k, build_gradient=True, build_hessian=True):
        self.k = np.array(k).ravel()
        K = np.diag(self.k)
        super(self.__class__, self).set_parameter(K, build_gradient, build_hessian)



