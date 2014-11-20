import numpy as np
seed = np.random.randint(2**16)
# seed = 50431
seed = 58482
print "Seed: ", seed

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from optofit.cneuron.compartment import Compartment, SquidCompartment
from optofit.cneuron.channels import LeakChannel, NaChannel, KdrChannel
from optofit.cneuron.simulate import forward_euler
from optofit.cneuron.gpchannel import GPChannel, sigma, GPKdrChannel

from hips.inference.particle_mcmc import *
from optofit.cinference.pmcmc import *

# Set the random seed for reproducibility
np.random.seed(seed)

# Make a simple compartment
hypers = {
            'C'      : 1.0,
            'V0'     : -60.0,
            'g_leak' : 0.03,
            'E_leak' : -65.0}

gp1_hypers = {'sig' : 1,
            'g_gp'   : 12.0,
            'g_gp'   : 0.0,
            'E_gp'   : 50.0}

gp2_hypers = { 'sig' : 1,
            'g_gp'   : 3.60,
            'E_gp'   : -77.0}

squid_hypers = {
            'C'      : 1.0,
            'V0'     : -60.0,
            'g_leak' : 0.03,
            'E_leak' : -65.0,
            'g_na'   : 12.0,
            'g_na'   : 0.0,
            'E_na'   : 50.0,
            'g_kdr'  : 3.60,
            'E_kdr'  : -77.0
         }


def create_gp_model():
    # Add a few channels
    body = Compartment(name='body', hypers=hypers)
    leak = LeakChannel(name='leak', hypers=hypers)
    gp1 = GPChannel(name='gpna', hypers=gp1_hypers)
    gp2 = GPKdrChannel(name='gpk', hypers=gp2_hypers)

    body.add_child(leak)
    body.add_child(gp1)
    body.add_child(gp2)
    # Initialize the model
    D, I = body.initialize_offsets()

    return body, gp1, gp2, D, I

def sample_squid_model():
    squid_body = SquidCompartment(name='body', hypers=squid_hypers)
    # squid_body = Compartment(name='body', hypers=squid_hypers)
    # leak = LeakChannel(name='leak', hypers=squid_hypers)
    # na = NaChannel(name='na', hypers=squid_hypers)
    # kdr = KdrChannel(name='kdr', hypers=squid_hypers)
    # squid_body.add_child(leak)
    # body.add_child(na)
    # squid_body.add_child(kdr)

    # Initialize the model
    D, I = squid_body.initialize_offsets()

    # Set the recording duration
    t_start = 0
    t_stop = 100.
    dt = 0.1
    t_ds = 0.1
    t = np.arange(t_start, t_stop, dt)
    T = len(t)

    # Make input with an injected current from 500-600ms
    inpt = np.zeros((T, I))
    inpt[20/dt:80/dt,:] = 7.
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

    # Downsample
    intvl = int(t_ds / dt)
    td = t[::intvl].copy('C')
    zd = z[::intvl, :].copy('C')
    xd = x[::intvl, :].copy('C')
    inptd = inpt[::intvl].copy('C')

    # Plot the first particle trajectory
    plt.ion()
    st_axs, _ = squid_body.plot(td, zd, color='k')
    # Plot the observed voltage
    st_axs[0].plot(td, xd[:,0], 'r')

    # plt.plot(t, x[:,0],  'r')
    plt.show()
    plt.pause(0.01)
    return td, zd, xd, inptd, st_axs

def sample_from_model(T,D, init, prop):
    z = np.zeros((T,1,D))
    z[0,0,:] = init.sample()[:]
    # Sample the latent state sequence with the given initial condition
    for i in np.arange(0,T-1):
        # The interface kinda sucks. We have to tell it that
        # the first particle is always its ancestor
        prop.sample_next(z, i, np.array([0], dtype=np.int32))

    return z[:,0,:]

# Now run the pMCMC inference
def sample_gp_given_true_z(t, x, inpt,
                           z_squid,
                           N_particles=100,
                           axs=None, gp1_ax=None, gp2_ax=None):
    dt = np.diff(t)
    T,O = x.shape

    # Make a model
    body, gp1, gp2, D, I = create_gp_model()

    # Set the initial distribution to be Gaussian around the steady state
    ss = np.zeros(D)
    body.steady_state(ss)
    init = GaussianInitialDistribution(ss, 0.1**2 * np.eye(D))

    # Set the proposal distribution using Hodgkin Huxley dynamics
    sigmas = 0.001*np.ones(D)
    # Set the voltage transition dynamics to be a bit noisier
    sigmas[body.x_offset] = 0.25
    prop = HodgkinHuxleyProposal(T, N_particles, D, body,  sigmas, t, inpt)

    # Set the observation model to observe only the voltage
    etas = np.ones(1)
    observed_dims = np.array([body.x_offset]).astype(np.int32)
    lkhd = PartialGaussianLikelihood(observed_dims, etas)

    # Initialize the latent state matrix with the equivalent of the squid latent state
    z = np.zeros((T,1,D))
    z[:,0,0] = z_squid[:,0]              # V = V

    m3h = z_squid[:,1]**3 * z_squid[:,2]
    m3h = np.clip(m3h, 1e-4,1-1e-4)
    z[:,0,1] = np.log(m3h/(1.0-m3h))     # Na open fraction

    n4 = z_squid[:,3]**4
    n4 = np.clip(n4, 1e-4,1-1e-4)
    z[:,0,2] = np.log(n4/(1.0-n4))       # Kdr open fraction


    # Prepare the particle Gibbs sampler with the first particle
    pf = ParticleGibbsAncestorSampling(T, N_particles, D)
    pf.initialize(init, prop, lkhd, x, z[:,0,:].copy('C'))

    # Plot the initial state
    gp1_ax, im1, l_gp1 = gp1.plot(ax=gp1_ax, data=z[:,0,:])
    gp2_ax, im2, l_gp2 = gp2.plot(ax=gp2_ax, data=z[:,0,:])
    axs, lines = body.plot(t, z[:,0,:], color='b', axs=axs)
    axs[0].plot(t, x[:,0], 'r')

    # Plot a sample from the model
    lpred = axs[0].plot(t, sample_from_model(T,D, init, prop)[:,0], 'g')

    # Update figures
    for i in range(1,4):
        plt.figure(i)
        plt.pause(0.001)

    # Initialize sample outputs
    S = 1000
    z_smpls = np.zeros((S,T,D))
    z_smpls[0,:,:] = z[:,0,:]

    for s in range(1,S):
        print "Iteration %d" % s
        # Reinitialize with the previous particle
        # pf.initialize(init, prop, lkhd, x, z_smpls[s-1,:,:])

        # Sample a new trajectory given the updated kinetics and the previous sample
        # z_smpls[s,:,:] = pf.sample()
        z_smpls[s,:,:] = z_smpls[s-1,:,:]

        # Resample the GP
        gp1.resample(z_smpls[s,:,:], dt)
        gp2.resample(z_smpls[s,:,:], dt)

        # Resample the conductances
        # resample_body(body,  t, z_smpls[s,:,:], sigmas[0])

        # Plot the sample
        body.plot(t, z_smpls[s,:,:], lines=lines)
        gp1.plot(im=im1, l=l_gp1, data=z_smpls[s,:,:])
        gp2.plot(im=im2, l=l_gp2, data=z_smpls[s,:,:])

        # Sample from the model and plot
        lpred[0].set_data(t, sample_from_model(T,D, init,prop)[:,0])

        # Update figures
        for i in range(1,4):
            plt.figure(i)
            plt.pause(0.001)

    z_mean = z_smpls.mean(axis=0)
    z_std = z_smpls.std(axis=0)
    z_env = np.zeros((T*2,2))

    z_env[:,0] = np.concatenate((t, t[::-1]))
    z_env[:,1] = np.concatenate((z_mean[:,0] + z_std[:,0], z_mean[::-1,0] - z_std[::-1,0]))

    plt.ioff()
    plt.show()

    return z_smpls

t, z, x, inpt, st_axs = sample_squid_model()
raw_input("Press enter to being sampling...\n")
sample_gp_given_true_z(t, x, inpt, z, axs=st_axs)


