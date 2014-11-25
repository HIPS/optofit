import copy
import cPickle

import numpy as np
seed = np.random.randint(2**16)
# seed = 2958
seed = 60017
print "Seed: ", seed



import matplotlib.pyplot as plt

from optofit.cneuron.compartment import Compartment, SquidCompartment
from optofit.cneuron.gpchannel import GPChannel

from hips.inference.particle_mcmc import *
from optofit.cinference.pmcmc import *

from hips.plotting.layout import  *

# Set the random seed for reproducibility
np.random.seed(seed)

# Make a simple compartment
hypers = {
            'C'      : 1.0,
            'V0'     : -60.0,
            'g_leak' : 0.03,
            'E_leak' : -65.0}

gp1_hypers = {'D': 2,
              'sig' : 1,
              'g_gp'   : 12.0,
              'E_gp'   : 50.0,
              'alpha_0': 1.0,
              'beta_0' : 2.0,
              'sigma_kernel': 1.0}

gp2_hypers = {'D' : 1,
              'sig' : 1,
              'g_gp'   : 3.60,
              # 'g_gp'   : 0,
              'E_gp'   : -77.0,
              'alpha_0': 1.0,
              'beta_0' : 2.0,
              'sigma_kernel': 1.0}

squid_hypers = {
            'C'      : 1.0,
            'V0'     : -60.0,
            'g_leak' : 0.03,
            'E_leak' : -65.0,
            'g_na'   : 12.0,
            # 'g_na'   : 0.0,
            'E_na'   : 50.0,
            'g_kdr'  : 3.60,
            'E_kdr'  : -77.0
         }


def create_gp_model():
    # Add a few channels
    body = Compartment(name='body', hypers=hypers)
    leak = LeakChannel(name='leak', hypers=hypers)
    gp1 = GPChannel(name='gpna', hypers=gp1_hypers)
    gp2 = GPChannel(name='gpk', hypers=gp2_hypers)

    body.add_child(leak)
    body.add_child(gp1)
    body.add_child(gp2)
    # Initialize the model
    D, I = body.initialize_offsets()

    return body, gp1, gp2, D, I

def sample_squid_model():
    squid_body = SquidCompartment(name='body', hypers=squid_hypers)

    # Initialize the model
    D, I = squid_body.initialize_offsets()

    # Set the recording duration
    t_start = 0
    t_stop = 300.
    dt = 0.1
    t = np.arange(t_start, t_stop, dt)
    T = len(t)

    # Make input with an injected current from 500-600ms
    inpt = np.zeros((T, I))
    inpt[20/dt:40/dt,:] = 3.
    inpt[120/dt:160/dt,:] = 5.
    inpt[220/dt:280/dt,:] = 7.
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
        prop.sample_next(z, i, np.zeros((N,), dtype=np.int32))

    # Sample observations
    for i in np.arange(0,T):
        lkhd.sample(z,x,i,0)

    # Extract the first (and in this case only) particle
    z = z[:,0,:].copy(order='C')

    # Downsample
    t_ds = 0.1
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

def make_figure_1(t, inpt, z_true, z_smpls, gpna_smpls, gpk_smpls):
    """
    Make figure 1.

    :param t:
    :param z_true:
    :param z_smpls:
    :param gpna_smpls:
    :param gpk_smpls:
    :return:
    """
    # Parse out the true latent states
    V_true = z_true[:,0]
    m_true = z_true[:,1]
    h_true = z_true[:,2]
    n_true = z_true[:,3]

    na_true = m_true**3 * h_true
    k_true = n_true**4

    # Extract the inferred states
    offset = 2
    z_mean = z_smpls[offset:,...].mean(0)
    z_std = z_smpls[offset:,...].mean(0)
    V_inf = z_mean[:,:,0]
    na_inf = z_mean[:,:,1]
    k_inf = z_mean[:,:,3]

    # Make the figure
    fig = create_figure((5.5,4))
    V_ax = create_axis_at_location(fig, 0.5, 3, 4.5, 0.75,
                                   transparent=True, box=False)
    V_ax.plot(t, V_true)




# Simulate the squid compartment to get the ground truth
t, z_true, x, inpt, st_axs = sample_squid_model()

# Load the results of the pMCMC inference
with open('squid_results.pkl', 'r') as f:
    z_smpls, gp1_smpls, gp2_smpls = cPickle.load(f)




