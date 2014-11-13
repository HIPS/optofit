import numpy as np
# Set the random seed for reproducibility
seed = np.random.randint(2**16)

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from optofit.cneuron.compartment import Compartment
from optofit.cneuron.channels import LeakChannel
from optofit.cneuron.simulate import forward_euler
from optofit.cneuron.gpchannel import GPChannel, sigma

from hips.inference.particle_mcmc import *
from optofit.cinference.pmcmc import *

# Make a simple compartment
hypers = {
            'C'      : 1.0,
            'V0'     : -60.0,
            'g_leak' : 0.3,
            'E_leak' : -65.0,
            'g_gp'   : 1.0,
            'E_gp'   : 0.0,
         }

print "Seed: ", seed
np.random.seed(seed)

def plot_state(t, z, axs=None, lines=None, I=None, color='k'):
    if lines is None and axs is None:

        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        l1 = ax1.plot(t, z[:,0], color=color)
        ax1.set_ylabel('V')

        ax2 = fig.add_subplot(312)
        l2 = ax2.plot(t, sigma(z[:,1]), color=color)
        ax2.set_ylabel('\\sigma(z_1)')
        ax2.set_ylim((0,1))

        ax3 = fig.add_subplot(313)
        l3 = None
        if I is not None:
            l3 = ax3.plot(t, I, color=color)
            ax3.set_ylabel('I_{gp}')
            ax3.set_xlabel('t')

        axs = [ax1, ax2, ax3]
        lines = [l1, l2, l3]

    elif lines is None and axs is not None:
        ax1 = axs[0]
        l1 = ax1.plot(t, z[:,0], color=color)

        ax2 = axs[1]
        l2 = ax2.plot(t, sigma(z[:,1]), color=color)

        ax3 = axs[2]
        l3 = None
        if I is not None:
            l3 = ax3.plot(t, I, color=color)

        lines = [l1, l2, l3]

    elif lines is not None:
        lines[0][0].set_data(t, z[:,0])
        lines[1][0].set_data(t, sigma(z[:,1]))
        if I is not None:
            lines[2][0].set_data(t, I)

    return axs, lines

def sample_model():
    # Add a few channels
    body = Compartment(name='body', hypers=hypers)
    leak = LeakChannel(name='leak', hypers=hypers)
    gp = GPChannel(name='gp', hypers=hypers)

    body.add_child(leak)
    body.add_child(gp)
    # Initialize the model
    D, I = body.initialize_offsets()

    # Set the recording duration
    t_start = 0
    t_stop = 100.
    dt = 1.0
    t = np.arange(t_start, t_stop, dt)
    T = len(t)

    # Make input with an injected current from 500-600ms
    inpt = np.zeros((T, I))
    inpt[50/dt:60/dt,:] = 7.
    inpt += np.random.randn(T, I)

    # Set the initial distribution to be Gaussian around the steady state
    z0 = np.zeros(D)
    body.steady_state(z0)
    init = GaussianInitialDistribution(z0, 0.1**2 * np.eye(D))

    # Set the proposal distribution using Hodgkin Huxley dynamics
    sigmas = 0.0001*np.ones(D)
    # Set the voltage transition dynamics to be a bit noisier
    sigmas[body.x_offset] = 0.25
    prop = HodgkinHuxleyProposal(T, 1, D, body,  sigmas, t, inpt)

    # Set the observation model to observe only the voltage
    etas = np.ones(1)
    observed_dims = np.array([body.x_offset]).astype(np.int32)
    lkhd = PartialGaussianLikelihood(observed_dims, etas)

    # Initialize the latent state matrix to sample N=1 particle
    z = np.zeros((T,1,D))
    z[0,0,:] = init.sample()
    # Initialize the output matrix
    x = np.zeros((T,D))

    # Sample the latent state sequence
    for i in np.arange(0,T-1):
        # The interface kinda sucks. We have to tell it that
        # the first particle is always its ancestor
        prop.sample_next(z, i, np.array([0], dtype=np.int32))

    # Sample observations
    for i in np.arange(0,T):
        lkhd.sample(z,x,i,0)

    # Extract the GP current
    I_gp = gp.current(z, z[:,0,0], np.arange(T), 0)

    # Extract the first (and in this case only) particle
    z = z[:,0,:].copy(order='C')

    # Plot the GP channel dynamics
    gp_fig = plt.figure()
    gp_ax1 = gp_fig.add_subplot(121)
    gp.plot(ax=gp_ax1)
    gp_ax2 = gp_fig.add_subplot(122)

    # Plot the first particle trajectory
    st_axs, _ = plot_state(t, z, I=I_gp, color='k')
    # Plot the observed voltage
    st_axs[0].plot(t, x[:,0], 'r')

    plt.ion()
    plt.show()
    plt.pause(0.01)

    return t, z, x, inpt, gp, st_axs, gp_ax2

# Now run the pMCMC inference
def sample_z_given_x(t, x, inpt, gp,
                     z0=None,
                     N_particles=100,
                     plot=False,
                     axs=None, gp_ax=None):

    T,O = x.shape

    # Make a model
    # Add a few channels
    body = Compartment(name='body', hypers=hypers)
    leak = LeakChannel(name='leak', hypers=hypers)
    gp = GPChannel(name='gp', hypers=hypers)

    body.add_child(leak)
    body.add_child(gp)
    # Initialize the model
    D, I = body.initialize_offsets()

    # Set the initial distribution to be Gaussian around the steady state
    ss = np.zeros(D)
    body.steady_state(ss)
    init = GaussianInitialDistribution(ss, 0.1**2 * np.eye(D))

    # Set the proposal distribution using Hodgkin Huxley dynamics
    sigmas = 0.0001*np.ones(D)
    # Set the voltage transition dynamics to be a bit noisier
    sigmas[body.x_offset] = 0.25
    prop = HodgkinHuxleyProposal(T, N_particles, D, body,  sigmas, t, inpt)

    # Set the observation model to observe only the voltage
    etas = np.ones(1)
    observed_dims = np.array([body.x_offset]).astype(np.int32)
    lkhd = PartialGaussianLikelihood(observed_dims, etas)

    # Initialize the latent state matrix to sample N=1 particle
    z = np.ones((T,N_particles,D)) * ss[None, None, :]
    if z0 is not None:
        z[:,0,:] = z0

    # Prepare the particle Gibbs sampler with the first particle
    pf = ParticleGibbsAncestorSampling(T, N_particles, D)
    pf.initialize(init, prop, lkhd, x, z[:,0,:].copy('C'))

    # Plot the initial state
    I_gp = gp.current(z, z[:,0,0], np.arange(T), 0)
    gp_ax, im = gp.plot(ax=gp_ax)

    axs, lines = plot_state(t, z[:,0,:], color='b', I=I_gp, axs=axs)

    # Update figures
    plt.figure(1)
    plt.pause(0.001)
    plt.figure(2)
    plt.pause(0.001)

    raw_input("Press any key to continue\n")

    # Initialize sample outputs
    S = 1
    z_smpls = np.zeros((S,T,D))

    for s in range(S):
        print "Iteration %d" % s
        # Reinitialize with the previous particle
        pf.initialize(init, prop, lkhd, x, z_smpls[s,:,:])

        # Sample a new trajectory given the updated kinetics and the previous sample
        z_smpls[s,:,:] = pf.sample()

        # Plot the sample
        I_gp = gp.current(z_smpls[s,:,:][:,None,:], z_smpls[s,:,0], np.arange(T), 0)
        plot_state(t, z_smpls[s,:,:], I=I_gp, lines=lines)
        plt.autoscale()

        # Update the latent state figure
        plt.figure(2)
        plt.pause(0.001)

        # Resample the GP
        import pdb; pdb.set_trace()
        # raw_input("Press any key to continue\n")
        gp.resample(z_smpls[s,:,:])
        gp.plot(im=im)

        # Update the gp transition figure
        plt.figure(1)
        plt.pause(0.001)

    z_mean = z_smpls.mean(axis=0)
    z_std = z_smpls.std(axis=0)
    z_env = np.zeros((T*2,2))

    z_env[:,0] = np.concatenate((t, t[::-1]))
    z_env[:,1] = np.concatenate((z_mean[:,0] + z_std[:,0], z_mean[::-1,0] - z_std[::-1,0]))

    plt.ioff()
    if plot:
        axs[0].add_patch(Polygon(z_env, facecolor='b', alpha=0.25, edgecolor='none'))

        # Compute the current
        plot_state(t, z_mean, axs=axs, color='b')

        # Plot a few random samples
        # for s in range(10):
        #     si = np.random.randint(S)
        #     plt.plot(t, z_smpls[si,:,0], '-b', lw=0.5)
        plt.pause(0.001)

    return z_smpls

t, z, x, inpt, gp, axs, gp_ax = sample_model()

raw_input("Press enter to being sampling...\n")
sample_z_given_x(t, x, inpt, gp, plot=True, axs=axs, gp_ax=gp_ax, z0=z)