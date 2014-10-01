"""
General code for plotting the state of the neuron
"""
from optofit.utils.utils import extract_names_from_dtype, get_item_at_path, sz_dtype, as_matrix
from optofit.observation.observable import *
import matplotlib.pyplot as plt
import numpy as np

def plot_latent_compartment_state(t, z, state, compartment, axs=None, colors=['k'], linewidth=1):
    dtype = compartment.latent_dtype
    lb = compartment.latent_lb
    ub = compartment.latent_ub
    D = sz_dtype(dtype)
    z_comp = get_item_at_path(z, compartment.path)
    z = as_matrix(z_comp, D)
    z_names = extract_names_from_dtype(dtype)

    # Compute the channel currents
    s_comp = get_item_at_path(state, compartment.path)
    N_ch = len(compartment.channels)
    Is = [s_comp[ch.name]['I'] for ch in compartment.channels]

    # if fig is None:
    #     fig,axs = plt.subplots(D,1)
    # else:
    #     axs = fig.get_axes()
    # # Make sure axs is a list of axes, even if it is length 1
    # if not isinstance(axs, (list, np.ndarray)):
    #     axs = [axs]


    if axs is None:
        axs = []

        for d in np.arange(D):
            ax = plt.subplot2grid((D,3), (d,0), colspan=2)
            axs.append(ax)

        ax = plt.subplot2grid((D,3), (0,2), rowspan=D)
        axs.append(ax)

    for d in np.arange(D):
        axs[d].plot(t, z[d,:], color=colors[d % len(colors)], linewidth=linewidth)
        axs[d].set_ylabel(z_names[d])

        yl = list(axs[d].get_ylim())
        if np.isfinite(lb[d]):
            yl[0] = lb[d]
        if np.isfinite(ub[d]):
            yl[1] = ub[d]
        axs[d].set_ylim(yl)

    # Plot the channel densities
    C = len(compartment.channels)
    gs = [ch.g.value for ch in compartment.channels]
    names = [ch.name for ch in compartment.channels]


    axs[-1].bar(np.arange(C), gs, facecolor=colors[0], alpha=0.5)
    axs[-1].set_xticks(np.arange(C))
    axs[-1].set_xticklabels(map(lambda n: '$g_%s$' % n, names))
    axs[-1].set_title('Channel densities')
    axs[-1].set_ylim([0,30])

    # if not fig_given:
    #     plt.show()

    return axs

def plot_latent_compartment_V_and_I(t, data_sequence, compartment, observation,
                                    axs=None, colors=['k'], linewidth=1):

    Z = data_sequence.latent
    S = data_sequence.states
    O = data_sequence.observations
    V = get_item_at_path(Z, compartment.path)['V']
    Ca = get_item_at_path(Z, compartment.path)['[Ca]']

    if isinstance(observation, NewDirectCompartmentVoltage):
        F = get_item_at_path(O, observation.path)['V']
    if isinstance(observation, LowPassCompartmentVoltage):
        F = get_item_at_path(O, observation.path)['V']
    elif isinstance(observation, LinearFluorescence):
        F = get_item_at_path(O, observation.path)['Flr']
    else:
        F = None

    # Check for inputs
    I = np.zeros_like(t)
    try:
        I = get_item_at_path(data_sequence.input, compartment.path)['I']
    except:
        # No input current
        pass

    try:
        I = get_item_at_path(data_sequence.input, compartment.path)['Irr']
    except:
        # No input irradiance
        pass

    # Compute the channel currents
    s_comp = get_item_at_path(S, compartment.path)

    # Num rows = N_ch (oner per channel current) +
    # 1 (input) + 1 (voltage) + 1 (calcium)
    D = len(compartment.channels) + 3

    # Set the relative width of the time series to the conductances
    r = 3

    if axs is None:
        axs = []
        for d in np.arange(D):
            ax = plt.subplot2grid((D,r+1), (d,0), colspan=r)
            axs.append(ax)

        # Add one more axis for the concentrations
        ax = plt.subplot2grid((D,r+1), (0,r), rowspan=D)
        axs.append(ax)
    
    # Plot the voltage)
    axs[0].plot(t, I, color='b', lw=linewidth)
    axs[0].set_ylabel('$I_{%s}$' % compartment.name)

    # Plot the voltage
    axs[1].plot(t, V, color=colors[0], lw=linewidth)
    axs[1].set_ylabel('$V_{%s}$' % compartment.name)

    # Plot the calcium
    axs[2].plot(t, Ca, color=colors[0], lw=linewidth)
    axs[2].set_ylabel('$[Ca]_{%s}$' % compartment.name)

    if F is not None:
        axs[1].plot(t, F, color='b', lw=linewidth)

    for i,ch in enumerate(compartment.channels):
        I = s_comp[ch.name]['I']
        axs[i+3].plot(t, I, color=colors[i % len(colors)], linewidth=linewidth)
        axs[i+3].set_ylabel('$I_{%s}$' % ch.name)

    # Plot the channel densities
    C = len(compartment.channels)
    gs = [ch.g.value for ch in compartment.channels]
    names = [ch.name for ch in compartment.channels]

    axs[-1].bar(np.arange(C), gs, facecolor=colors[0], alpha=0.5)
    axs[-1].set_xticks(0.5+np.arange(C))
    axs[-1].set_xticklabels(map(lambda n: '$g_{%s}$' % n, names))
    axs[-1].set_title('Channel densities')

    return axs


def plot_latent_state(t, z, dtype, fig=None, colors=['k'], linewidth=1):
    D,T = z.shape
    z_names = extract_names_from_dtype(dtype)

    plt.ion()
    fig_given = fig is not None
    if not fig_given:
        fig,axs = plt.subplots(D,1)
    else:
        axs = fig.get_axes()

    # Make sure axs is a list of axes, even if it is length 1
    if not isinstance(axs, list):
        axs = [axs]

    for d in np.arange(D):
        axs[d].plot(t, z[d,:], color=colors[d % len(colors)], linewidth=linewidth)
        axs[d].set_ylabel(z_names[d])

    if not fig_given:
        plt.show()

def plot_latent_currents(t, z, neuron, inpt, fig=None, colors=['k']):
    state = neuron.evaluate_state(z, inpt)
    T = z.size
    # Get the latent currents
    I_names = []
    gs = np.array([])
    Is = np.zeros((T,0))
    for c in neuron.compartments:
        # Get the sub-structured arrays for this comp
        chs = c.channels

        # Now compute the per channel currents in this compartment
        for ch in chs:
            if state[c.name][ch.name].dtype.names is not None and \
               'I' in state[c.name][ch.name].dtype.names:
                gs = np.concatenate((gs, [ch.g.value]))
                Is = np.concatenate((Is, -1.0*state[c.name][ch.name]['I'][:,np.newaxis]),
                                    axis=1)
                I_names.append('I_'+ch.name)

    C = Is.shape[1]

    fig_given = fig is not None
    if not fig_given:
        fig,axs = plt.subplots(C,1)
    else:
        axs = fig.get_axes()

    # Make sure axs is a list of axes, even if it is length 1
    # if not isinstance(axs, np.ndarray):
    #     axs = np.array([axs])

    for c in np.arange(C):
        axs[c].plot(t, Is[:,c], color=colors[c % len(colors)], linewidth=2)
        axs[c].set_ylabel(I_names[c])

    if not fig_given:
        plt.show()

    return fig, axs
