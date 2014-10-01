import numpy as np
# Set the random seed for reproducibility
seed = np.random.randint(2**16)
print "Seed: ", seed
np.random.seed(seed)

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from optofit.cneuron.compartment import Compartment, SquidCompartment
from optofit.cneuron.channels import LeakChannel, NaChannel, KdrChannel
from optofit.cneuron.simulate import forward_euler

from hips.inference.particle_mcmc import *
from optofit.cinference.pmcmc import *

# Make a simple compartment
hypers = {
            'C'      : 1.0,
            'V0'     : -60.0,
            'g_leak' : 0.3,
            'E_leak' : -65.0,
            'g_na'   : 120.0,
            'E_na'   : 50.0,
            'g_kdr'  : 36.0,
            'E_kdr'  : -77.0
         }

def sample_model():
    # # Add a few channels
    # body = Compartment(name='body', hypers=hypers)
    # leak = LeakChannel(name='leak', hypers=hypers)
    # na = NaChannel(name='na', hypers=hypers)
    # kdr = KdrChannel(name='kdr', hypers=hypers)
    #
    # body.add_child(leak)
    # body.add_child(na)
    # body.add_child(kdr)
    # Initialize the model
    # body.initialize_offsets()

    squid_body = SquidCompartment(name='body', hypers=hypers)

    # Initialize the model
    D, I = squid_body.initialize_offsets()

    # Set the recording duration
    t_start = 0
    t_stop = 100.
    dt = 0.01
    t = np.arange(t_start, t_stop, dt)
    T = len(t)

    # Make input with an injected current from 500-600ms
    inpt = np.zeros((T, I))
    inpt[50/dt:60/dt,:] = 7.
    inpt += np.random.randn(T, I)

    # Set the initial distribution to be Gaussian around the steady state
    z0 = np.zeros(D)
    squid_body.steady_state(z0)
    init = GaussianInitialDistribution(z0, 0.1**2 * np.eye(D))

    # Set the proposal distribution using Hodgkin Huxley dynamics
    # TODO: Fix the hack which requires us to know the number of particles
    N = 100
    sigmas = 0.0001*np.ones(D)
    # Set the voltage transition dynamics to be a bit noisier
    sigmas[squid_body.x_offset] = 0.25
    prop = HodgkinHuxleyProposal(T, N, D, squid_body,  sigmas, t, inpt)

    # Set the observation model to observe only the voltage
    etas = np.ones(1)
    observed_dims = np.array([squid_body.x_offset]).astype(np.int32)
    lkhd = PartialGaussianLikelihood(observed_dims, etas)

    # Initialize the latent state matrix to sample N=1 particle
    z = np.zeros((T,N,D))
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

    # Extract the first (and in this case only) particle
    z = z[:,0,:].copy(order='C')

    # Plot the first particle trajectory
    plt.ion()
    fig = plt.figure()
    # fig.add_subplot(111, aspect='equal')
    plt.plot(t, z[:,observed_dims[0]], 'k')
    plt.plot(t, x[:,0],  'r')
    plt.show()

    return t, z, x, init, prop, lkhd

# Now run the pMCMC inference
def sample_z_given_x(t, z_curr, x,
                     init, prop, lkhd,
                     N_particles=100,
                     plot=False):

    T,D = z_curr.shape
    T,O = x.shape
    # import pdb; pdb.set_trace()
    pf = ParticleGibbsAncestorSampling(T, N_particles, D)
    pf.initialize(init, prop, lkhd, x, z_curr)

    S = 10
    z_smpls = np.zeros((S,T,D))
    for s in range(S):
        print "Iteration %d" % s
        # Reinitialize with the previous particle
        pf.initialize(init, prop, lkhd, x, z_smpls[s,:,:])
        z_smpls[s,:,:] = pf.sample()

    z_mean = z_smpls.mean(axis=0)
    z_std = z_smpls.std(axis=0)
    z_env = np.zeros((T*2,2))

    z_env[:,0] = np.concatenate((t, t[::-1]))
    z_env[:,1] = np.concatenate((z_mean[:,0] + z_std[:,0], z_mean[::-1,0] - z_std[::-1,0]))

    if plot:
        plt.gca().add_patch(Polygon(z_env, facecolor='b', alpha=0.25, edgecolor='none'))
        plt.plot(t, z_mean[:,0], 'b', lw=1)


        # Plot a few random samples
        # for s in range(10):
        #     si = np.random.randint(S)
        #     plt.plot(t, z_smpls[si,:,0], '-b', lw=0.5)

        plt.ioff()
        plt.show()

    return z_smpls

t, z, x, init, prop, lkhd = sample_model()
sample_z_given_x(t, z, x, init, prop, lkhd, plot=True)