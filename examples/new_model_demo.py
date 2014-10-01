import numpy as np
# Set the random seed for reproducibility
seed = np.random.randint(2**16)
print "Seed: ", seed
np.random.seed(seed)

import matplotlib.pyplot as plt

from optofit.models.model import Model
from optofit.population.population import Population
from optofit.neuron.neuron import Neuron
from optofit.neuron.compartment import Compartment, CalciumCompartment
from optofit.neuron.channels import *
from optofit.simulation.stimulus import PeriodicStepStimulusPattern, DirectCompartmentCurrentInjection, DirectCompartmentIrradiance
from optofit.simulation.simulate import simulate
from optofit.observation.observable import NewDirectCompartmentVoltage, LowPassCompartmentVoltage, IndependentObservations, LinearFluorescence
from optofit.plotting.plotting import plot_latent_compartment_state, plot_latent_compartment_V_and_I
from optofit.inference.fitting import fit_mcmc


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
    # body_voltage = NewDirectCompartmentVoltage('body voltage', model, body)
    # observation.add_observation(body_voltage)

    # Low pass filtered voltage measurement
    lp_body_voltage = LowPassCompartmentVoltage('lp body voltage', model, body, filterbins=2)
    observation.add_observation(lp_body_voltage)

    # Fluorescence (linearly scaled voltage) measurement
    # body_fluorescence = LinearFluorescence('body fluorescence' , model, body)
    # observation.add_observation(body_fluorescence)

    model.add_observation(observation)

    return model

# Instantiate the true model
true_model = make_model()

# Create a stimulus for the neuron
# Stimulate the neuron by injecting a current pattern
stim_on = 10
stim_off = np.Inf
stim_on_dur = 100
stim_off_dur = 100.
stim_I = 5.
stim_pattern = PeriodicStepStimulusPattern(stim_on, stim_off, stim_on_dur, stim_off_dur, stim_I)
# stim = DirectCompartmentCurrentInjection(true_model.population.neurons[0].compartments[0], stim_pattern)
stim = DirectCompartmentIrradiance(true_model.population.neurons[0].compartments[0], stim_pattern)
# Set the recording duration
t_start = 0
t_stop = 100.
dt = 0.01
t = np.arange(t_start, t_stop, dt)

# Also make a model that we will use for inference
inferred_model = make_model()

# Make a set of data sequences ('experiments')
N_experiments = 1
for e in range(N_experiments):
    # Simulate the model to create synthetic data
    data_sequence = simulate(true_model, t, stim)
    true_model.add_data_sequence(data_sequence)

    # Make a corresponding data sequence for the inference model
    inf_data_sequence = simulate(inferred_model, t, stim)
    inf_data_sequence.observations = data_sequence.observations

    # Condition on the observed voltage
    inferred_model.add_data_sequence(inf_data_sequence)

# Specify whether to plot the true and observed voltage
# seq_to_plot = 0
# plot = 'states'
plot = 'currents'
plt.ion()
i = {'i' : 0}
# Add a callback to update the plots
def plot_sample(m):
    for seq_to_plot in range(N_experiments):
        fig = plt.figure(seq_to_plot)
        fig.clf()
        if plot == 'states':
            axs = plot_latent_compartment_state(t,
                                                true_model.data_sequences[seq_to_plot].latent,
                                                true_model.data_sequences[seq_to_plot].states,
                                                true_model.population.neurons[0].compartments[0])
            plot_latent_compartment_state(t,
                                          m.data_sequences[seq_to_plot].latent,
                                          m.data_sequences[seq_to_plot].states,
                                          m.population.neurons[0].compartments[0],
                                          axs=axs, colors=['r'])
        elif plot == 'currents':
            axs = plot_latent_compartment_V_and_I(t,
                                                  true_model.data_sequences[seq_to_plot],
                                                  true_model.population.neurons[0].compartments[0],
                                                  true_model.observation.observations[0])
            plot_latent_compartment_V_and_I(t,
                                            m.data_sequences[seq_to_plot],
                                            m.population.neurons[0].compartments[0],
                                            m.observation.observations[0],
                                          axs=axs, colors=['r'])
        fig.suptitle('Iteration: %d' % i['i'])
    i['i'] += 1
    plt.pause(0.1)

# Plot the initial sample
plot_sample(inferred_model)

# Generic fitting code will enumerate the components of the model and determine
# which MCMC updates to use.
raw_input("Press enter to begin MCMC")
print "Running particle MCMC"
samples = fit_mcmc(inferred_model, 20, callback=plot_sample)
