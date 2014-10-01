import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from pybiophys.models.hyperparameters import hypers
from pybiophys.models.model import point_parameter_model, DataSequence
from pybiophys.neuron.channels       import *
from pybiophys.observation.observable import NewDirectCompartmentVoltage, LowPassCompartmentVoltage, IndependentObservations, LinearFluorescence

def pickle_model(model, filename):
    pickle.dump(model_to_dict(model), open(filename, 'w'))

def model_to_dict(model):
    return {
        'observable': observation_to_dict(model.observation),
        'data': [data_sequence_to_dict(ds) for ds in model.data_sequences],
        'neuron': compartment_to_dict(model.population.neurons[0].compartments[0])
    }

def observation_to_dict(observation):
    return {'sigma': observation.observations[0].sigma.value}

def channel_to_dict(channel):
    return (channel.name, {'g': channel.g.value, 'E': channel.E.value})

def compartment_to_dict(compartment):
    channels = dict([channel_to_dict(ch) for ch in compartment.channels] )
    return channels

def data_sequence_to_dict(ds):
    return {
        'latent':       ds.latent,
        'observations': ds.observations,
        'input':        ds.input,
        'states':       ds.states
    }

def conductances_from_name(samples, name):
    return np.array([s['neuron'][name]['g'] for s in samples])

def model_dict_to_model(model_dict):
    model, body = point_parameter_model(model_dict['neuron'])
    observation = IndependentObservations('observations', model)
    lp_body_voltage = LowPassCompartmentVoltage('lp body voltage', model, body, filterbins=20)
    lp_body_voltage.sigma.value = model_dict['observable']['sigma']
    observation.add_observation(lp_body_voltage)
    model.add_observation(observation)
    
    for ds in model_dict['data']:
        t = .1 * np.array(range(len(ds['input'])))
        model.add_data_sequence(DataSequence(None, t, ds['input'], ds['observations'], ds['latent'], ds['input'], ds['states']))

    return model

def hist(true, samples, name, burn=20, ax = None):
    if not ax:
        plt.hist(conductances_from_name(samples[burn:], name))
        plt.axvline(true['neuron'][name]['g'], color='r')
        plt.show()
    else:
        data = conductances_from_name(samples[burn:], name)
        ax.hist(data, 30, normed=1)
        ax.axvline(true['neuron'][name]['g'], color='r')
        x = np.linspace(np.min(data), np.max(data))
        rv = gamma(hypers['a_g_' + name].value, scale = 1 / hypers['b_g_' + name].value)
        ax.plot(x, rv.pdf(x), color='g')

import os.path
def zip_files(filename):
    seed, true, samples = pickle.load(open(filename + "_1.pk", 'r'))
    last = samples[-1]
    i = 2
    while os.path.isfile(filename + "_" + str(i) + ".pk"):
        _, _, next_samples = pickle.load(open(filename + "_" + str(i) + ".pk", 'r'))
        if not np.allclose(next_samples[0]['data'][0]['states'].view(np.float64), last['data'][0]['states'].view(np.float64)):
            print "ERROR: first and last of the file aren't the same"
        samples = samples + next_samples[1:]
        last = next_samples[-1]
        i += 1
    return seed, true, samples
        
def path_trace(true, samples, name, ax = None):
    if not ax:
        plt.axhline(true['neuron'][name]['g'], color='r')
        data = conductances_from_name(samples, name)
        plt.plot(range(len(data)), data)
        plt.show()
    else:
        ax.axhline(true['neuron'][name]['g'], color='r')
        data = conductances_from_name(samples, name)
        ax.plot(range(len(data)), data)

def plot_all(plot_fun, true, samples, burn = 20, names = ['leak', 'ca3kdr', 'ca3ka', 'ca3na', 'ca3ca', 'ca3kahp', 'ca3kc', 'chr2']):
    channels = names
    f, axes = plt.subplots(len(channels))
    for ax, name in zip(axes, channels):
        plot_fun(true, samples[burn:], name, ax)
        ax.set_ylabel(name)
    plt.show()

def percentile_plot(true, samples, name, ax):
    #print samples[0]['data']
    #import pdb; pdb.set_trace()
    data = np.array([s['data'][0]['latent']['neuron']['body'][name] for s in samples])
    #print data
    #import pdb; pdb.set_trace()
    top    = np.percentile(data, 97.5, axis=0)
    bottom = np.percentile(data, 2.5, axis=0)
    mean   = np.mean(data, axis = 0)
    t      = range(len(mean))
    ax.plot(t, mean, color = 'r')
    ax.fill_between(t, top, bottom, facecolor="teal")
    ax.plot(t, true['data'][0]['latent']['neuron']['body'][name], color='b')
    
