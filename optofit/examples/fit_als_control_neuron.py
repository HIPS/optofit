import os
import numpy as np
from scipy.io import loadmat

# Set the random seed for reproducibility
seed = np.random.randint(2**16)
print "Seed: ", seed
np.random.seed(seed)

import matplotlib.pyplot as plt

from pybiophys.models.model import Model, DataSequence
from pybiophys.population.population import Population
from pybiophys.neuron.neuron import Neuron
from pybiophys.neuron.compartment import Compartment, CalciumCompartment
from pybiophys.simulation.simulate import simulate
from pybiophys.neuron.channels import *
from pybiophys.simulation.stimulus import GivenStimulusPattern, DirectCompartmentCurrentInjection, DirectCompartmentIrradiance
from pybiophys.observation.observable import NewDirectCompartmentVoltage, LowPassCompartmentVoltage, IndependentObservations, LinearFluorescence
from pybiophys.plotting.plotting import plot_latent_compartment_state, plot_latent_compartment_V_and_I
from pybiophys.inference.fitting import fit_mcmc


def make_model():
    """Make a model of a single compartment neuron with a handful of channels and a directly
    observable voltage
    """
    model = Model()
    # The population object doesn't do anything yet, but eventually it could support
    # synapses between neurons
    population = Population('population', model)
    # Explicitly build the neuron
    neuron = Neuron('neuron', population)
    # The single compartment corresponds to the cell body
    body = CalciumCompartment('body', neuron)
    # Add a few channels
    leak = LeakChannel('leak', body)

    ca3kdr = Ca3KdrChannel('ca3kdr', body)
    ca3ka = Ca3KaChannel('ca3ka', body)
    ca3na = Ca3NaChannel('ca3na', body)
    ca3ca = Ca3CaChannel('ca3ca', body)
    ca3kahp = Ca3KahpChannel('ca3kahp', body)
    ca3kc = Ca3KcChannel('ca3kc', body)
    chr2 = ChR2Channel('chr2', body)

    # Now connect all the pieces of the neuron together
    body.add_channel(leak)
    # body.add_channel(na)
    # body.add_channel(kdr)

    body.add_channel(ca3kdr)
    body.add_channel(ca3ka)
    body.add_channel(ca3na)
    body.add_channel(ca3ca)
    body.add_channel(ca3kahp)
    body.add_channel(ca3kc)
    body.add_channel(chr2)

    neuron.add_compartment(body, None)
    population.add_neuron(neuron)
    model.add_population(population)

    # Create the observation model
    observation = IndependentObservations('observations', model)
    # Direct voltage measurement
    body_voltage = NewDirectCompartmentVoltage('body voltage', model, body)
    observation.add_observation(body_voltage)

    # # Low pass filtered voltage measurement
    # lp_body_voltage = LowPassCompartmentVoltage('lp body voltage', model, body, filterbins=2)
    # observation.add_observation(lp_body_voltage)

    # Fluorescence (linearly scaled voltage) measurement
    # body_fluorescence = LinearFluorescence('body fluorescence' , model, body)
    # observation.add_observation(body_fluorescence)

    model.add_observation(observation)

    return model

def load_data(T_start=None, T_stop=None):
    """
    Load a sample of data from the control experiment
    :return:
    """
    datafile = os.path.join('data', 'fv_comp', 'fov2-2_data.mat')
    datamat = loadmat(datafile)

    t_V = datamat['t_V'].astype(np.float).ravel()
    V = datamat['V'].astype(np.float).ravel()
    stim = datamat['blue'].astype(np.float).ravel()
    t_stim = t_V

    t_F = datamat['t_F'].astype(np.float).ravel()
    F = datamat['F'].astype(np.float).ravel()

    sf_V = datamat['sample_rate_V'].astype(np.int)
    sf_F = datamat['sample_rate_F'].astype(np.int)
    sf_stim = sf_V

    # Extract only the given range of data
    if T_start is not None and T_stop is not None:
        V = V[np.bitwise_and(t_V >= T_start, t_V <= T_stop)]
        # Stimulus is given at the same frequency as the voltage
        stim = stim[np.bitwise_and(t_V >= T_start, t_V <= T_stop)]
        # Fluorescence is given at a slower sampling frequency
        F = F[np.bitwise_and(t_F >= T_start, t_F <= T_stop)]

        # Segment up the time indices
        t_V = t_V[np.bitwise_and(t_V >= T_start, t_V <= T_stop)]
        t_stim = t_V
        t_F = t_F[np.bitwise_and(t_F >= T_start, t_F <= T_stop)]

    return {'V' : V,
            't_V' : t_V,
            'sample_rate_V' : sf_V,
            'stim' : stim,
            't_stim' : t_stim,
            'sample_rate_stim' : sf_stim,
            'F' : F,
            't_F' : t_F,
            'sample_rate_F' : sf_F}

def initialize_model(data):
    """
    Fit the ALS data
    """
    # Make a model that we will use for inference
    model = make_model()
    # Create a stimulus for the neuron
    stim_pattern = GivenStimulusPattern(data['t_stim'], data['stim'], data['sample_rate_V'])
    stim = DirectCompartmentIrradiance(model.population.neurons[0].compartments[0], stim_pattern)

    # Condition on the observed voltage
    obs = np.zeros((len(data['t_V']),), dtype=model.observation.observed_dtype)
    obs['body voltage']['V'] = data['V']

    # Add the data sequence to the model
    data_sequence = simulate(model, data['t_V'], stim)
    data_sequence.observations = obs
    # model.add_data('seq1', data['t_V'], stim, obs)
    model.add_data_sequence(data_sequence)

    return model

def fit_model(model, data):
    # Specify whether to plot the true and observed voltage
    # seq_to_plot = 0
    # plot = 'states'
    plot = 'currents'
    plt.ion()
    i = {'i' : 0}
    # Add a callback to update the plots
    def plot_sample(m):
        fig = plt.figure(0)
        fig.clf()
        if plot == 'states':
            axs = plot_latent_compartment_state(data['t_V'],
                                          m.data_sequences[0].latent,
                                          m.data_sequences[0].states,
                                          m.population.neurons[0].compartments[0],
                                          colors=['r'])

            # Plot the true voltage
            axs[1].plot(data['t_V'], data['V'])

        elif plot == 'currents':
            axs = plot_latent_compartment_V_and_I(data['t_V'],
                                            m.data_sequences[0],
                                            m.population.neurons[0].compartments[0],
                                            m.observation.observations[0],
                                            colors=['r'])
            # Plot the true voltage
            axs[1].plot(data['t_V'], data['V'])

        fig.suptitle('Iteration: %d' % i['i'])
        i['i'] += 1
        plt.pause(0.1)

    # Plot the initial sample
    plot_sample(model)

    # Generic fitting code will enumerate the components of the model and determine
    # which MCMC updates to use.
    raw_input("Press enter to begin MCMC")
    print "Running particle MCMC"
    samples = fit_mcmc(model, 2, callback=plot_sample)


data = load_data(T_start=11, T_stop=11.05)
model = initialize_model(data)
fit_model(model, data)