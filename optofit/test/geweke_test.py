__author__ = 'scott'
import numpy as np
# Set the random seed for reproducibility
seed = np.random.randint(2**16)
print "Seed: ", seed
np.random.seed(seed)



import matplotlib.pyplot as plt

from pybiophys.models.model import Model
from pybiophys.population.population import Population
from pybiophys.neuron.neuron import Neuron
from pybiophys.neuron.compartment import Compartment, CalciumCompartment
from pybiophys.neuron.channels import LeakChannel, NaChannel, KdrChannel, Ca3KdrChannel, Ca3KaChannel, Ca3NaChannel, Ca3CaChannel, Ca3KahpChannel, Ca3KcChannel
from pybiophys.simulation.stimulus import PeriodicStepStimulusPattern, DirectCompartmentCurrentInjection
from pybiophys.simulation.simulate import simulate
from pybiophys.observation.observable import NewDirectCompartmentVoltage, IndependentObservations, LinearFluorescence
from pybiophys.plotting.plotting import plot_latent_compartment_state, plot_latent_compartment_V_and_I
from pybiophys.inference.fitting import fit_mcmc
from pybiophys.models.hyperparameters import hypers


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
    body = Compartment('body', neuron)
    # body = CalciumCompartment('body', neuron)
    # Add a few channels
    # body.add_channel(LeakChannel('leak', body))
    # body.add_channel(NaChannel('na', body))
    body.add_channel(KdrChannel('kdr', body))


    # ca3kdr = Ca3KdrChannel('ca3kdr', body)
    # ca3ka = Ca3KaChannel('ca3ka', body)
    # ca3na = Ca3NaChannel('ca3na', body)
    # ca3ca = Ca3CaChannel('ca3ca', body)
    # ca3kahp = Ca3KahpChannel('ca3kahp', body)
    # ca3kc = Ca3KcChannel('ca3kc', body)
    #
    #body.add_channel(ca3kdr)
    #body.add_channel(ca3ka)
    #body.add_channel(ca3na)
    #body.add_channel(ca3ca)
    #body.add_channel(ca3kahp)
    #body.add_channel(ca3kc)

    # Now connect all the pieces of the neuron together
    neuron.add_compartment(body, None)
    population.add_neuron(neuron)
    model.add_population(population)

    # Create the observation model
    observation = IndependentObservations('observations', model)
    body_voltage = NewDirectCompartmentVoltage('body voltage', model, body)
    # body_fluorescence = LinearFluorescence('body fluorescence' , model, body)
    # observation.add_observation(body_fluorescence)
    observation.add_observation(body_voltage)
    model.add_observation(observation)

    return model

# Instantiate the true model
true_model = make_model()

# Create a stimulus for the neuron
# Stimulate the neuron by injecting a current pattern
stim_on = 2.0
stim_off = 50.0
stim_on_dur = .5
stim_off_dur = 1.5
stim_I = 500.0
stim_pattern = PeriodicStepStimulusPattern(stim_on, stim_off, stim_on_dur, stim_off_dur, stim_I)
stim = DirectCompartmentCurrentInjection(true_model.population.neurons[0].compartments[0], stim_pattern)

# Set the recording duration
t_start = 0
t_stop = 0.2
dt = 0.1
t = np.arange(t_start, t_stop, dt)

# Simulate the model to create synthetic data
data_sequence = simulate(true_model, t, stim)
true_model.add_data_sequence(data_sequence)


# Plot the true and observed voltage
plt.ion()
fig = plt.figure(figsize=(8,6))
# axs = plot_latent_compartment_state(t, z, true_model.population.neurons[0].compartments[0])
axs = plot_latent_compartment_V_and_I(t, data_sequence,
                                      true_model.population.neurons[0].compartments[0],
                                      true_model.observation.observations[0],)

i = {'i' : 0}
# Add a callback to update the plots

def plot_sample(m):
    plt.gcf().clf()
    # latent = m.data_sequences[0].latent
    # plot_latent_compartment_state(t, m.data_sequences[0].latent,
    #                               m.data_sequences[0].states,
    #                               m.population.neurons[0].compartments[0])
    axs = plot_latent_compartment_V_and_I(t, m.data_sequences[0],
                                          m.population.neurons[0].compartments[0],
                                          m.observation.observations[0])
    print '%d: g_leak: %f' % (i['i'], m.population.neurons[0].compartments[0].channels[0].g.value)
    print '%d: g_na: %f' % (i['i'], m.population.neurons[0].compartments[0].channels[1].g.value)
    print '%d: g_kdr: %f' % (i['i'], m.population.neurons[0].compartments[0].channels[2].g.value)

    fig.suptitle('Iteration: %d' % i['i'])
    i['i'] += 1
    plt.pause(0.001)

def print_g_leak(m):
    if np.mod(i['i'], 1) == 0:
        # print '%d: g_leak: %f' % (i['i'], m.population.neurons[0].compartments[0].channels[0].g.value)
        # print '%d: g_na: %f' % (i['i'], m.population.neurons[0].compartments[0].channels[1].g.value)
        print '%d: g_kdr: %f' % (i['i'], m.population.neurons[0].compartments[0].channels[0].g.value)
    i['i'] += 1

# Generic fitting code will enumerate the components of the model and determine
# which MCMC updates to use.
raw_input("Press enter to begin MCMC")
print "Running particle MCMC"

# samples = fit_mcmc(true_model, N_samples=1000, callback=plot_sample, geweke=True)
samples = fit_mcmc(true_model, N_samples=10000, callback=print_g_leak, print_interval=10, geweke=True)


# Plot the results
import scipy.stats
def plot_channel(samples, index, name, a, b, xlim=None):
    gs = np.array([m.population.neurons[0].compartments[0].channels[index].g.value for m in samples])
    plt.figure()
    _,bins,_ = plt.hist(gs, 50, normed=True, alpha=0.5)

    if xlim is None:
        plt.plot(bins, scipy.stats.gamma.pdf(bins, a, scale=b))
    else:
        xx = np.linspace(xlim[0], xlim[1])
        plt.plot(xx, scipy.stats.gamma.pdf(xx, a, scale=1.0/b))
    plt.title('$g_{%s}' % name)

# plot_channel(samples, 0, 'leak', hypers['a_g_leak'].value, hypers['b_g_leak'].value, (1e-4,3))
# plot_channel(samples, 1, 'na', hypers['a_g_na'].value, hypers['b_g_na'].value, (1,30))
plot_channel(samples, 0, 'kdr', hypers['a_g_kdr'].value, hypers['b_g_kdr'].value, (1,14))


plt.ioff()
plt.show()
