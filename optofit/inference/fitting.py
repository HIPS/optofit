"""
Holds the overall fitting code.
"""
import copy
import numpy as np

from mcmc_transitions import initialize_updates
from pybiophys.simulation.simulate import simulate
from pybiophys.utils.utils import as_matrix

def fit_mcmc(model, N_samples=1000, callback=None, print_interval=1, geweke=False):
    """
    Fit the parameters of hte model with MCMC

    :param model:
    :param data:
    :return:
    """
    # Get a list of MCMC updates for the model
    updates = initialize_updates(model, geweke)

    # Sample an initial state of the model from the prior
    samples = []

    initialize_model(model)
    samples.append(model)

    # Collect samples
    for i in range(1,N_samples):
        if np.mod(i, print_interval) == 0:
            print "Iteration: %d" % i
        curr_model = copy.deepcopy(samples[i-1])

        # Call the callback
        if callback is not None:
            try:
                callback(curr_model)
            except Exception as e:
                print "WARNING: Caught exception during callback:"
                print e

        # Go through each update
        for update in updates:
            update.update(curr_model)

        # Debug:
        # curr_states = curr_model.data_sequences[0].states
        # prev_states = samples[-1].data_sequences[0].states
        # d_states = (as_matrix(curr_states) - as_matrix(prev_states)).mean()
        # print 'd_states: ', d_states
        #
        # curr_latent = curr_model.data_sequences[0].latent
        # prev_latent = samples[-1].data_sequences[0].latent
        # d_latent = (as_matrix(curr_latent) - as_matrix(prev_latent)).mean()
        # print 'd_latent: ', d_latent

        samples.append(curr_model)

    return samples

def initialize_model(model):
    """
    Find a decent parameter regime to start the model
    Most of the model parameters have been drawn from the prior, so
    we'll just focus on the latent states

    :param model:
    :return:
    """
    for data in model.data_sequences:
        if np.allclose(as_matrix(data.latent), 0.0) or \
           np.allclose(as_matrix(data.states), 0.0):
            z, _ = simulate(model, data.t, data.stimuli)
            data.latent = z
            data.states = model.population.evaluate_state(data.latent, data.input)