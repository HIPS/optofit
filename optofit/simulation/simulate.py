"""
Simulate a neuron with hodgkin huxley dynamics
"""
import numpy as np

from optofit.simulation.stimulus import *
from optofit.models.model import DataSequence
from optofit.utils.utils import as_sarray
from optofit.models.hyperparameters import hypers
from optofit.inference.distributions import TruncatedGaussianDistribution

def simulate(model, t, stimuli, name=None, have_noise=True, I=None):
    """
    Simulate a model containing a population of neurons, an observation model,
    and a stimulus.

    Right now this only is set up for a single neuron model

    :param model:
    :return:
    """
    # TODO: Iterate over populations and neurons
    assert model.population is not None, "Models population must be instantiated"
    population = model.population
    #import pdb; pdb.set_trace()
    # Make sure stimuli is a list
    if isinstance(stimuli, Stimulus):
        stimuli = [stimuli]
    else:
        assert isinstance(stimuli, list) or I is not None

    T = len(t)
    if I is None:
        I = np.zeros((T,), dtype=population.input_dtype)
        for stim in stimuli:
            stim.set_input(t, I)

    # Set up the initial conditions with steady state values
    z0 = population.steady_state()
    # Create a function to implement hodgkin huxley dynamics
    def _hh_dynamics(z_vec, ti):
        # Set the inputs
        inpt = I[ti]
        #import pdb; pdb.set_trace()
        latent = as_sarray(z_vec, population.latent_dtype)
        # state = as_sarray(population.evaluate_state(latent, inpt), population.state_dtype)
        dzdt = population.kinetics(latent, inpt)
        return dzdt.view(np.float)

    # Do forward euler by hand.
    z0 = z0.view(np.float)
    D = len(z0)
    z = np.zeros((D,T))
    z[:,0] = z0
    lb = population.latent_lb
    ub = population.latent_ub

    if have_noise:
        noisedistribution = TruncatedGaussianDistribution()
        # TODO: Fix this hack
        sig_trans = np.ones(1, dtype=population.latent_dtype)
        sig_trans.fill(np.asscalar(hypers['sig_ch'].value))
        for neuron in population.neurons:
            for compartment in neuron.compartments:
                sig_trans[neuron.name][compartment.name]['V'] = hypers['sig_V'].value
                sig_trans = sig_trans.view(dtype=np.float)

    for ti in np.arange(1,T):
        dt = t[ti]-t[ti-1]
        z[:,ti] = z[:,ti-1] + dt * _hh_dynamics(z[:,ti-1], ti-1)
        z[:,ti] = np.clip(z[:,ti], lb, ub)

        if have_noise:
            # Add Gaussian noise
            noise_lb = lb - z[:,ti]
            noise_ub = ub - z[:,ti]

            # Scale down the noise to account for time delay
            # sig = sig_trans * np.sqrt(dt)
            sig = sig_trans * dt

            # TODO: Remove the zeros_like requirement
            noise = noisedistribution.sample(mu=np.zeros_like(z[:,ti]), sigma=sig,
                                         lb=noise_lb, ub=noise_ub)
            # import pdb; pdb.set_trace()
            z[:,ti] += noise

    # Convert back to numpy struct array
    Z = as_sarray(z, population.latent_dtype)

    # Compute the corresponding states
    S = population.evaluate_state(Z, I)

    # sample the observation model given the latent variables of the neuron
    O = model.observation.sample(Z)

    # Package into a data sequence
    ds = DataSequence(name, t, stimuli, O, Z, I, S)

    return ds

def recreate(model, t, I):
    # TODO: Iterate over populations and neurons
    assert model.population is not None, "Models population must be instantiated"
    population = model.population
    T = len(t)

    # Set up the initial conditions with steady state values
    z0 = population.steady_state()
    # Create a function to implement hodgkin huxley dynamics
    def _hh_dynamics(z_vec, ti):
        # Set the inputs
        inpt = I[ti]
        latent = as_sarray(z_vec, population.latent_dtype)
        # state = as_sarray(population.evaluate_state(latent, inpt), population.state_dtype)
        dzdt = population.kinetics(latent, inpt)
        return dzdt.view(np.float)

    # Do forward euler by hand.
    z0 = z0.view(np.float)
    D = len(z0)
    z = np.zeros((D,T))
    z[:,0] = z0
    lb = population.latent_lb
    ub = population.latent_ub

    for ti in np.arange(1,T):
        dt = t[ti]-t[ti-1]
        z[:,ti] = z[:,ti-1] + dt * _hh_dynamics(z[:,ti-1], ti-1)
        z[:,ti] = np.clip(z[:,ti], lb, ub)

    # Convert back to numpy struct array
    Z = as_sarray(z, population.latent_dtype)

    # Compute the corresponding states
    S = population.evaluate_state(Z, I)

    # sample the observation model given the latent variables of the neuron
    O = model.observation.sample(Z)

    # Package into a data sequence
    ds = DataSequence(name, t, stimuli, O, Z, I, S)

    return ds
