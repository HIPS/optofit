"""
When we're doing optogenetics, we don't actually get to directly observe
voltages, instead we just get to see flourescences. As such, we need to account
for the mapping from Voltage to Fluorescence.

The fluorescence is a linear transformation of the voltage (modulo effects at the
extremes of the range), coupled with an exponential decay as the proteins break 
down.

In addition, we don't actually observe Fluorescence in real-time. We have a fixed
sampling rate, which may or may not correspond to the time steps that we take 
while simulating the neuron.

The basic idea of this part of the program is to take in a "naive" neuron
compartment (one which assumes that we directly observe the voltages), and give 
us a new compartment which has Voltage as a latent and state
"""

import numpy as np
from pybiophys.utils import utils
from pybiophys.models.component import Component
from pybiophys.models.parameters import Parameter
from pybiophys.models.hyperparameters import hypers
from pybiophys.inference.distributions import GaussianDistribution, InverseGammaDistribution

class Observation(Component):
    """
    Superclass for different observations
    """
    def __init__(self, name, model):
        super(Observation, self).__init__()
        self._name = name
        self._observed_dtype = None
        self.model = model

    @property
    def observed_dtype(self):
        return self._observed_dtype

    @observed_dtype.setter
    def observed_dtype(self, value):
        self._observed_dtype = value

    def sample(self, data_sequence):
        """
        Sample a set of observations for the given data sequence
        """
        raise NotImplementedError()

    def logp(self, latent, observations):
        """
        Compute the log probability of observing this data sequence
        """
        raise NotImplementedError()

class IndependentObservations(Observation):
    """
    Class for a product of independent observations
    """
    def __init__(self, name, model):
        """

        :param name:
        :param model:
        :return:
        """
        super(IndependentObservations, self).__init__(name, model)

        # Initialize a list of observations
        self.observations = []
        self.observed_dtype = []

    # TODO: HACK: The dtypes are laid out such that the population is the root, but
    # the path property would make the model the root. Override path property
    # to work around this issue
    @property
    def path(self):
        if self.parent is None:
            return []

    def add_observation(self, observation):
        """
        Add an observation to the model

        :param neuron: The neuronto be added
        :return:
        """
        self.observations.append(observation)

        # Make a description of the latent state variables
        # Each set of latent variables is accessed by the neuron name
        if observation.observed_dtype is not None:
            self._observed_dtype.append((observation.name, observation.observed_dtype))

    def logp(self, latent, observations):
        """
        Compute the log probability of these observations

        :param data_sequence:
        :return:
        """
        logp = 0
        for observation in self.observations:
            logp += observation.logp(latent, observations)

        return logp

    def sample(self, latent):
        """
        Compute the log probability of these observations

        :param data_sequence:
        :return:
        """
        o = np.zeros(latent.shape, dtype=self.observed_dtype)
        for observation in self.observations:
            o[observation.name] = observation.sample(latent)

        return o

class NewDirectCompartmentVoltage(Observation):
    """
    Slight variation on the current direct compartment voltage observation.
    """
    def __init__(self, name, model, compartment):
        super(NewDirectCompartmentVoltage, self).__init__(name, model)
        self.compartment = compartment
        self.sigma = Parameter('sigma', 5.0)

        # Set the datatype of this observation
        self.observed_dtype = [('V', np.float64)]

    def sample(self, latent):
        """
        Sample an observable voltage given the latent state
        :param latent_state:
        :return:
        """
        o = np.zeros(latent.shape, self.observed_dtype)
        x_comp = utils.get_item_at_path(latent, self.compartment.path)
        V = x_comp['V']

        o['V'] = V + self.sigma.value * np.random.randn(*V.shape)
        return o

    def logp(self, latent, observations):
        """
        Compute the log probability of the observations given the latent states

        :param latent:
        :param observations:
        :return:
        """
        x_comp = utils.get_item_at_path(latent, self.compartment.path)
        latent_V = x_comp['V']

        o = utils.get_item_at_path(observations, self.path)
        observed_V = o['V']

        logp = -0.5/self.sigma.value**2 * (observed_V-latent_V)**2
        return logp

class CompartmentObservation(Observation):
    def __init__(self, name, model, compartment):
        super(CompartmentObservation, self).__init__(name, model)
        self.compartment     = compartment

    def get_my(self, inpt):
        if not self.name in inpt.dtype.names:
            return utils.get_item_at_path(inpt, self.compartment.path)
        else:
            return inpt[self.name]

    def sample(self, data_sequence):
        latent = self.get_my(data_sequence)
        return self._sample(latent)

    def logp(self, latent, observations):
        latent       = self.get_my(latent)
        observations = self.get_my(observations)
        return self._logp(latent, observations)

    def update(self, latent, observations):
        latent = self.get_my(latent)
        observations = self.get_my(observations)
        self._update(latent, observations)

class LinearFluorescence(CompartmentObservation):
    def __init__(self, name, model, compartment, sigma = None, theta = None):
        super(LinearFluorescence, self).__init__(name, model, compartment)
        self._observed_dtype = [('Flr', np.float64)]
        self._input_dtype    = [('V', np.float64)]

        self._sigma_sq = sigma
        self._theta = theta
        if sigma is None:
            self._sigma_sq = Parameter('sigma squared', distribution = InverseGammaDistribution(3, 3))
        if theta is None:
            self._theta = Parameter(
                'theta',
                np.array([[1], [0]]),
                distribution = GaussianDistribution(2, self._sigma_sq.value * np.array([[1], [0]]), np.identity(2))
            )
            self._ln_inv = np.identity(2)
        else:
            self._ln_inv = theta.distribution.cov

    @property
    def input_dtype(self):
        return self._input_dtype

    @property
    def scale(self):
        return self._theta.value[0][0]

    @property
    def intercept(self):
        return self._theta.value[1][0]

    @property
    def sigma(self):
        return np.sqrt(self._sigma_sq.value[0])

    def _sample(self, latent):
        out = self._transform(latent)
        out['Flr'] = out['Flr'] + self.sigma * np.random.randn(*out['Flr'].shape)
        return out

    def _logp(self, latent, observations):
        diffs = observations['Flr'] - self._transform(latent)['Flr']
        return (-0.5/(self.sigma ** 2)) * (diffs ** 2)

    def _transform(self, latent):
        o = np.zeros(latent['V'].shape, self.observed_dtype)
        o['Flr'] = latent['V'] * self.scale + self.intercept
        return o

    def _invert(self, observed):
        l = np.zeros(observed['Flr'].shape, self.input_dtype)
        l['V'] = (observed['Flr'] - self.intercept) / self.scale
        return l

    def _update(self, latent, observed):
        # The covariance seems suspect -- I keep drawing really big values
        v = latent['V']
        X = np.zeros((v.shape[0], 2))
        X[:, 0] = v
        X[:, 1] = np.ones(v.shape)
        y = np.matrix([observed['Flr']]).transpose()

        a, b, mu, ln_inv = update_normal_gamma(
            self._theta.distribution, self._sigma_sq.distribution,
            X, y
        )

        self._ln_inv   = ln_inv
        self._sigma_sq = Parameter('sigma squared', distribution = InverseGammaDistribution(a, b))
        self._theta    = Parameter('theta', distribution = GaussianDistribution(2, mu, self._sigma_sq.value.tolist()[0][0] *ln_inv))
        

    def _sample_params(self):
        sigma_sq = self._sigma_sq.distribution.sample()
        theta = GaussianDistribution(2, self._theta.distribution.mu, sigma_sq * self._ln_inv).sample()
        return sigma_sq, theta


class LowPassCompartmentVoltage(Observation):
    """
    Noisy measurement of a low pass filtered (averaged) compartment voltage
    """
    def __init__(self, name, model, compartment,
                 filterbins=1):
        """

        :param name:
        :param model:
        :param compartment:
        :param filterbins: Number of time bins to average over
        :return:
        """
        super(LowPassCompartmentVoltage, self).__init__(name, model)
        self.compartment = compartment
        self.sigma = hypers['sig_obs_V']

        assert isinstance(filterbins, int)
        self.filterbins = filterbins

        # Set the datatype of this observation
        self.observed_dtype = [('V', np.float64)]

    def sample(self, latent, noise=True):
        """
        Sample an observable voltage given the latent state
        :param latent_state:
        :return:
        """
        o = np.zeros(latent.shape, self.observed_dtype)
        x_comp = utils.get_item_at_path(latent, self.compartment.path)
        V = x_comp['V']
        assert V.ndim == 1, "V should be 1d"

        # Reshape into chunks of size filterbins
        V2 = V.reshape((-1,self.filterbins,))
        assert V2[0,1] == V[1], "Reshape failed!"

        # Compute the average
        Vlp = V2.sum(axis=1)/np.float(self.filterbins)

        if noise:
            # Add noise to the average
            Vlp += self.sigma.value * np.random.randn(*Vlp.shape)

        # Reshape into the size of V
        Vlp = np.repeat(Vlp, self.filterbins)
        Vlp = Vlp[:len(V)]

        o['V'] = Vlp
        return o

    def simple_logp(self, latent, observations):
        """
        We want to compute the log probability of a set of observations given
        the latent trajectories up to time t. This is challenging (especially
        in our current particle MCMC framework, since we are not set up for
        observation models that depend on previous as well as current states),
        so for now we'll just do something simple as an approximation.

        :param latent:
        :param observations:
        :return:
        """
        x_comp = utils.get_item_at_path(latent, self.compartment.path)
        latent_V = x_comp['V']

        o = utils.get_item_at_path(observations, self.path)
        observed_V = o['V']

        # Set the effective variance to be 'filterbins' times the variance
        # of the mean. This is motivated by thinking of each instantaneous
        # voltage measurement as an i.i.d. sample whose mean is the measured
        # fluorescence
        sigma_eff = np.sqrt(self.filterbins) * self.sigma.value
        logp = -0.5/sigma_eff**2 * (observed_V-latent_V)**2
        return logp

    def logp(self, latent, observations):
        return self.simple_logp(latent, observations)

def update_normal_gamma(normal, invgamma, x, y):
    # Uses Bayesian Linear Regression to update a normal inverse gamma distribution
    l0 = np.matrix(normal.cov)
    m0 = np.matrix(normal.mu)
    N  = x.shape[0]

    ln = np.dot(x.transpose(), x) + l0
    lninv = np.linalg.inv(ln)
    mn = lninv * (l0 * m0 + np.dot(x.transpose(), y))
    
    #print "Prev: ", a, b
    alpha = invgamma.a + .5 * N
    beta  = invgamma.b + .5 * (np.dot(y.transpose(), y) + m0.T * l0 * m0 - mn.T * ln * mn)
    #print "Cur: ", alpha, beta
    #print sigma.tolist()[0][0]

    #diffs = np.linalg.lstsq(x, y)[0] - mn
    #print "Gaussian Mean Error: ", np.dot(diffs.transpose(), diffs)

    return alpha, beta, mn, lninv
