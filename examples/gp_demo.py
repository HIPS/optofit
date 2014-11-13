import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from optofit.cneuron.compartment import Compartment
from optofit.cneuron.channels import LeakChannel
from optofit.cneuron.simulate import forward_euler
from optofit.cneuron.gpchannel import GPChannel, sigma

from hips.inference.particle_mcmc import *
from optofit.cinference.pmcmc import *

# Set the random seed for reproducibility
seed = np.random.randint(2**16)
seed = 10312
print "Seed: ", seed
np.random.seed(seed)

# Make a simple compartment
hypers = {
            'C'      : 1.0,
            'V0'     : -60.0,
            'g_leak' : 0.3,
            'E_leak' : -65.0,
            'g_gp'   : 1.0,
            'E_gp'   : 0.0,
         }

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
    st_axs, _ = body.plot(t, z, color='k')
    # Plot the observed voltage
    st_axs[0].plot(t, x[:,0], 'r')

    plt.ion()
    plt.show()
    plt.pause(0.01)

    return t, z, x, inpt, st_axs, gp_ax2

# Now run the pMCMC inference
def sample_z_given_x(t, x, inpt,
                     z0=None,
                     N_particles=100,
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
    z = np.ones((T,N_particles,D)) * ss[None, None, :] + np.random.randn(T,N_particles,D) * sigmas[None, None, :]
    if z0 is not None:
        z[:,0,:] = z0

    # Prepare the particle Gibbs sampler with the first particle
    pf = ParticleGibbsAncestorSampling(T, N_particles, D)
    pf.initialize(init, prop, lkhd, x, z[:,0,:].copy('C'))

    # Plot the initial state
    I_gp = gp.current(z, z[:,0,0], np.arange(T), 0)
    gp_ax, im, l_gp = gp.plot(ax=gp_ax, data=z[:,0,:])

    axs, lines = body.plot(t, z[:,0,:], color='b', axs=axs)

    # Update figures
    plt.figure(1)
    plt.pause(0.001)
    plt.figure(2)
    plt.pause(0.001)

    # Initialize sample outputs
    S = 100
    z_smpls = np.zeros((S,T,D))
    z_smpls[0,:,:] = z[:,0,:]

    for s in range(1,S):
        print "Iteration %d" % s
        # Reinitialize with the previous particle
        pf.initialize(init, prop, lkhd, x, z_smpls[s-1,:,:])

        # Sample a new trajectory given the updated kinetics and the previous sample
        z_smpls[s,:,:] = pf.sample()

        # Plot the sample
        body.plot(t, z_smpls[s,:,:], lines=lines)

        # Update the latent state figure
        plt.figure(2)
        plt.pause(0.001)

        # Resample the GP
        gp.resample(z_smpls[s,:,:])
        gp.plot(im=im, l=l_gp, data=z_smpls[s,:,:])

        # Update the gp transition figure
        plt.figure(1)
        plt.pause(0.001)

    z_mean = z_smpls.mean(axis=0)
    z_std = z_smpls.std(axis=0)
    z_env = np.zeros((T*2,2))

    z_env[:,0] = np.concatenate((t, t[::-1]))
    z_env[:,1] = np.concatenate((z_mean[:,0] + z_std[:,0], z_mean[::-1,0] - z_std[::-1,0]))

    plt.ioff()
    plt.show()

    return z_smpls

t, z, x, inpt, axs, gp_ax = sample_model()

raw_input("Press enter to being sampling...\n")
sample_z_given_x(t, x, inpt, axs=axs, gp_ax=gp_ax, z0=None)