import os
import sys
import copy
import cPickle

import numpy as np
seed = np.random.randint(2**16)
# seed = 2958
# seed = 60017

if "DISPLAY" not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

from optofit.cneuron.compartment import Compartment, SquidCompartment
from optofit.cneuron.channels import LeakChannel, NaChannel, KdrChannel
from optofit.cneuron.simulate import forward_euler
from optofit.cneuron.gpchannel import GPChannel, sigma

from hips.inference.particle_mcmc import *
from optofit.cinference.pmcmc import *

import kayak
import scipy


plot_progress = True

args = iter(sys.argv)
for line in args:
    if line == "--seed":
        seed = int(next(args))
    elif line == "--no_graph":
        plot_progress = False

# Set the random seed for reproducibility
np.random.seed(seed)
print "Seed: ", seed
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

def sample_squid_model(start = 20, stop = 80, intensity = 7.):
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
    t_stop = 600.
    dt = 0.1
    t = np.arange(t_start, t_stop, dt)
    T = len(t)

    inpt = np.zeros((T, I))
    inpt[20/dt:40/dt,:] = 3.
    inpt[120/dt:160/dt,:] = 5.
    inpt[220/dt:280/dt,:] = 7.
    inpt[300/dt:380/dt,:] = 9.
    inpt[500/dt:599/dt,:] = 11.
    
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

    st_axs = None
    if(plot_progress):
        # Plot the first particle trajectory
        plt.ion()
        st_axs, _ = squid_body.plot(td, zd, color='k')
        # Plot the observed voltage
        st_axs[0].plot(td, xd[:,0], 'r')

        # plt.plot(t, x[:,0],  'r')
        plt.show()
        plt.pause(0.01)

    return td, zd, xd, inptd, st_axs

def sample_gp_model():
    body, gp1, gp2, D, I = create_gp_model()
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

    # Extract the first (and in this case only) particle
    z = z[:,0,:].copy(order='C')

    st_axs = None
    if(plot_progress):
        # Plot the first particle trajectory
        st_axs, _ = body.plot(t, z, color='k')
        # Plot the observed voltage
        st_axs[0].plot(t, x[:,0], 'r')

        # Plot the GP channel dynamics
        # gp1_fig = plt.figure()
        # gp1_ax1 = gp1_fig.add_subplot(121)
        # gp1.plot(ax=gp1_ax1)
        # gp1_ax2 = gp1_fig.add_subplot(122)
        #
        # gp2_fig = plt.figure()
        # gp2_ax1 = gp2_fig.add_subplot(121)
        # gp2.plot(ax=gp2_ax1)
        # gp2_ax2 = gp2_fig.add_subplot(122)

        plt.ion()
        plt.show()
        plt.pause(0.01)

    return t, z, x, inpt, st_axs

# Now run the pMCMC inference
def sample_z_given_x(t, x, inpt,
                     z0=None,
                     initialize='constant',
                     N_particles=1000,
                     N_samples=100,
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
    # sigmas = np.ones(D)
    sigmas = 0.2*np.ones(D)
    # Set the voltage transition dynamics to be a bit noisier
    # sigmas[body.x_offset] = 0.25
    prop = HodgkinHuxleyProposal(T, N_particles, D, body,  sigmas, t, inpt)

    # Set the observation model to observe only the voltage
    etas = np.ones(1)
    observed_dims = np.array([body.x_offset]).astype(np.int32)
    lkhd = PartialGaussianLikelihood(observed_dims, etas)

    # Initialize the latent state matrix to sample N=1 particle
    z = np.ones((T,N_particles,D)) * ss[None, None, :] + np.random.randn(T,N_particles,D) * sigmas[None, None, :]

    if z0 is not None:
        if initialize == 'ground_truth':
            logit = lambda zz: np.log(zz/(1-zz))
            # Fix the observed voltage
            z[:, 0, body.x_offset] = z0[:, 0]
            # Fix the Na latent state
            m = z0[:,1]
            h = z0[:,2]
            z[:,0, gp1.x_offset] = logit(np.clip(m**3 *h, 1e-4,1-1e-4))
            # Fix the Kdr latent state
            n = z0[:,3]
            z[:,0, gp2.x_offset] = logit(np.clip(n**4, 1e-4, 1-1e-4))
        else:
            z[:,0,:] = z0
    elif initialize == 'from_model':
        # Sample the latent state sequence with the given initial condition
        for i in np.arange(0,T-1):
            # The interface kinda sucks. We have to tell it that
            # the first particle is always its ancestor
            prop.sample_next(z, i, np.array([0], dtype=np.int32))

            # Fix the observed voltage
            z[i+1, 0, body.x_offset] = x[i+1, body.x_offset]
    elif initialize == 'optimize':
        # By default, optimize the latent state
        # Set the voltage...
        z[:, 0, body.x_offset] = x[:, body.x_offset]
        # Set the initial latent trace
        z[1:, 0, 1:] = initial_latent_trace(body, inpt, x[:, 0], t).transpose()
        # Set the initial voltage
        z[0, 0, 1:]  = np.array([0, 0, 0])
    else:
        # Constant initialization
        pass

    # Initialize conductance values with MCMC to match the observed voltage...
    # body.resample(t, z[:,0,:])
    # resample_body(body, t, z[:,0,:], sigmas[0])
    #
    # if z0 is None:
    #     # Sample the latent state sequence with the given initial condition
    #     for i in np.arange(0,T-1):
    #         # The interface kinda sucks. We have to tell it that
    #         # the first particle is always its ancestor
    #         prop.sample_next(z, i, np.array([0], dtype=np.int32))

    # Resample the Gaussian processes
    # gp1.resample(z[:,0,:], dt)
    # gp2.resample(z[:,0,:], dt)

    # Prepare the particle Gibbs sampler with the first particle
    pf = ParticleGibbsAncestorSampling(T, N_particles, D)
    pf.initialize(init, prop, lkhd, x, z[:,0,:].copy('C'))

    if (plot_progress):
        # Plot the initial state
        gp1_ax, im1, l_gp1 = gp1.plot(ax=gp1_ax, data=z[:,0,:])
        gp2_ax, im2, l_gp2 = gp2.plot(ax=gp2_ax, data=z[:,0,:])
        axs, lines = body.plot(t, z[:,0,:], color='b', axs=axs)
        axs[0].plot(t, x[:,0], 'r')

        # Update figures
        for i in range(1,4):
            plt.figure(i)
            plt.pause(0.001)

    # Initialize sample outputs
    z_smpls = np.zeros((N_samples,T,D))
    z_smpls[0,:,:] = z[:,0,:]
    gp1_smpls = []
    gp2_smpls = []
    # Resample observation noise
    # eta_sqs = resample_observation_noise(z_smpls[0,:,:], x)
    # lkhd.set_etasq(eta_sqs)


    for s in range(1,N_samples):
        print "Iteration %d" % s
        # raw_input("Press enter to continue\n")
        # Reinitialize with the previous particle
        pf.initialize(init, prop, lkhd, x, z_smpls[s-1,:,:])

        # Sample a new trajectory given the updated kinetics and the previous sample
        z_smpls[s,:,:] = pf.sample()
        # z_smpls[s,:,:] = z_smpls[s-1,:,:]
        # print "dz: ", (z_smpls[s,:,:] - z_smpls[s-1,:,:]).sum(0)

        # Resample the GP
        gp1.resample(z_smpls[s,:,:], dt)
        gp2.resample(z_smpls[s,:,:], dt)

        # Resample the noise levels
        sigmasq = resample_transition_noise(body, z_smpls[s,:,:], inpt, t)
        # HACK: Fix the voltage transition noise
        # sigmasq[0] = 0.5
        print "Sigmasq: ", sigmasq
        # prop.set_sigmasq(sigmasq)
        gp1.set_sigmas(sigmasq)
        gp2.set_sigmas(sigmasq)
        # gp1.resample_transition_noise(z_smpls[s, :, :], t)
        # gp2.resample_transition_noise(z_smpls[s, :, :], t)

        # eta_sqs = resample_observation_noise(z_smpls[s,:,:], x)
        # lkhd.set_etasq(eta_sqs)

        # Resample the conductances
        # resample_body(body,  t, z_smpls[s,:,:], sigmas[0])

        if(plot_progress):
            # Plot the sample
            body.plot(t, z_smpls[s,:,:], lines=lines)
            gp1.plot(im=im1, l=l_gp1, data=z_smpls[s,:,:])
            gp2.plot(im=im2, l=l_gp2, data=z_smpls[s,:,:])

            # Update figures
            for i in range(1,4):
                plt.figure(i)
                plt.pause(0.001)

            gp1_smpls.append(gp1.gps)
            gp2_smpls.append(gp2.gps)

        freq = 1
        if s % freq == 0:
            with open('squid' + str(seed) + '_results' + str(s / freq) + '.pkl', 'w') as f:
                cPickle.dump((z_smpls, gp1_smpls, gp2_smpls), f, protocol=-1)
            if(s / freq > 1):
                os.remove('squid' + str(seed) + '_results' + str((s / freq) - 1) + '.pkl')
    z_mean = z_smpls.mean(axis=0)
    z_std = z_smpls.std(axis=0)
    z_env = np.zeros((T*2,2))

    z_env[:,0] = np.concatenate((t, t[::-1]))
    z_env[:,1] = np.concatenate((z_mean[:,0] + z_std[:,0], z_mean[::-1,0] - z_std[::-1,0]))

    if(plot_progress):
        plt.ioff()
        plt.show()

    return z_smpls, gp1_smpls, gp2_smpls

def resample_transition_noise(body, data, inpt, t,
                              alpha0=100, beta0=100):
    """
    Resample sigma, the transition noise variance, under an inverse gamma prior
    """
    # import pdb; pdb.set_trace()
    Xs = []
    X_preds = []
    X_diffs = []

    T = data.shape[0]
    D = data.shape[1]
    dxdt = np.zeros((T,1,D))
    x = np.zeros((T,1,D))
    x[:,0,:] = data

    # Compute kinetics of the voltage
    body.kinetics(dxdt, x, inpt, np.arange(T-1).astype(np.int32))
    dt = np.diff(t)
    # TODO: Loop over data
    dX_pred = dxdt[:-1, 0, :]
    dX_data = (data[1:, :] - data[:-1, :]) / dt[:,None]
    X_diffs = dX_pred - dX_data

    # Resample transition noise.
    X_diffs = np.array(X_diffs)
    n = X_diffs.shape[0]

    sigmasq = np.zeros(D)
    for d in range(D):
        alpha = alpha0 + n / 2.0
        beta  = beta0 + np.sum(X_diffs[:,d] ** 2) / 2.0
        # self.sigmas[d] = beta / alpha
        sigmasq[d] = 1.0 / np.random.gamma(alpha, 1.0/beta)

    # print "Sigma V: %.3f" % (sigmas[d])
    return sigmasq

def resample_observation_noise(z, x,
                               alpha0=1.0, beta0=1.0):
    """
    Resample sigma, the transition noise variance, under an inverse gamma prior
    """
    # TODO: Iterate over obs dimensions. For now assume 1d
    V_pred = z[:,0]
    V_data = x[:,0]
    V_diff = V_pred - V_data

    # Resample transition noise.
    n = V_diff.shape[0]

    alpha = alpha0 + n / 2.0
    beta  = beta0 + np.sum(V_diff ** 2) / 2.0
    etasq = 1.0 / np.random.gamma(alpha, 1.0/beta)

    print "eta V: %.3f" % (etasq)

    return np.array([etasq])


from hips.inference.mh import mh
def resample_body(body, ts=[], datas=[], sigma=1.0):
        """
        Resample the conductances of this neuron.
        """
        assert isinstance(datas, list) or isinstance(datas, np.ndarray)
        if isinstance(datas, np.ndarray):
            datas = [datas]

        if isinstance(ts, np.ndarray):
            ts = [ts]

        Is = []
        dV_dts = []
        # Compute I and dV_dt for each dataset
        for t,data in zip(ts, datas):
            # Compute dV dt
            T = data.shape[0]
            V = data[:,body.x_offset]
            dV_dt = (V[1:] - V[:-1])/(t[1:] - t[:-1])
            dV_dts.append(dV_dt[:,None])

            # Compute the (unscaled) currents through each channel
            I = np.empty((T-1, len(body.children)))
            for m,c in enumerate(body.children):
                for i in range(T-1):
                    I[i,m] = c.current(data[:,None,:].copy('C'), V[i], i, 0)
            Is.append(I)

        # Concatenate values from all datasets
        dV_dt = np.vstack(dV_dts)
        I = np.vstack(Is)

        # Now do a nonnegative regression of dVdt onto I
        gs = 0.1 * np.ones(len(body.children))
        perm = np.random.permutation(len(body.children))

        # Define a helper function to compute the log likelihood and make MH proposals
        def _logp(m, gm):
            gtmp = gs.copy()
            gtmp[m] = gm
            dV_dt_pred = I.dot(gtmp)
            return (-0.5/sigma * (dV_dt_pred - dV_dt)**2).sum()

        # Define a metropolis hastings proposal
        def _q(x0, xf):
            lx0, lxf = np.log(x0), np.log(xf)
            return -0.5 * (lx0-lxf)**2

        def _sample_q(x0):
            lx0 = np.log(x0)
            xf = np.exp(lx0 + np.random.randn())
            return xf

        # Sample each channel in turn
        for m in perm:
            gs[m] = mh(gs[m], lambda g: _logp(m, g), _q, _sample_q, steps=10)[-1]

        for c,g in zip(body.children, gs):
            c.g = g

        print "Gs: ", gs

def initial_latent_trace(body, inpt, voltage, t):
    I_true = np.diff(voltage) * body.C
    T      = I_true.shape[0]
    gs     = np.diag([c.g for c in body.children])
    D      = int(sum([c.D for c in body.children]))

    driving_voltage = np.dot(np.ones((len(body.children), 1)), np.array([voltage]))[:, :T]

    child_i      = 0
    for i in range(D):
        driving_voltage[i, :] = voltage[:T] - body.children[child_i].E

    
    
    K = np.array([[max(i-j, 0) for i in range(T)] for j in range(T)])
    K = K.T + K
    K = -1*(K ** 2)
    K = np.exp(K / 2)
    
    L = np.linalg.cholesky(K + (1e-7) * np.eye(K.shape[0]))
    Linv = scipy.linalg.solve_triangular(L.transpose(), np.identity(K.shape[0]))

    N = 1
    batch_size = 5000
    learn = .0000001
    runs = 10000

    batcher = kayak.Batcher(batch_size, N)
    
    inputs  = kayak.Parameter(driving_voltage)
    targets = kayak.Targets(np.array([I_true]), batcher)
    
    g_params       = kayak.Parameter(gs)
    I_input        = kayak.Parameter(inpt.T[:, :T])
    Kinv           = kayak.Parameter(np.dot(Linv.transpose(), Linv))

    initial_latent = np.random.randn(D, T)
    latent_trace   = kayak.Parameter(initial_latent)
    sigmoid        = kayak.Logistic(latent_trace)

    quadratic = kayak.ElemMult(
        sigmoid,
        kayak.MatMult(
            kayak.Parameter(np.array([[0, 1, 0],
                                      [0, 0, 0],
                                      [0, 0, 0]])),
            sigmoid
        )
    )
    three_quadratic = kayak.MatMult(
        kayak.Parameter(np.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 0]])),
        quadratic
    )
    linear = kayak.MatMult(
        kayak.Parameter(np.array([[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 1]])),
        sigmoid
    )
    
    leak_open      = kayak.Parameter(np.vstack((np.ones((1, T)), np.ones((2, T)))))
    open_fractions = kayak.ElemAdd(leak_open, kayak.ElemAdd(three_quadratic, linear))

    I_channels = kayak.ElemMult(
        kayak.MatMult(g_params, inputs),
        open_fractions
    )

    I_ionic   = kayak.MatMult(
        kayak.Parameter(np.array([[1, 1, 1]])),
        I_channels
    )

    predicted = kayak.MatAdd(I_ionic, I_input)

    nll = kayak.ElemPower(predicted - targets, 2)
          
    hack_vec = kayak.Parameter(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]))
    kyk_loss = kayak.MatSum(nll) + kayak.MatMult(
        kayak.Reshape(
            kayak.MatMult(
                kayak.MatMult(latent_trace, Kinv),
                kayak.Transpose(latent_trace)
            ),
            (9,)
        ),
        hack_vec
    ) + kayak.MatSum(kayak.ElemPower(I_channels, 2))

    grad = kyk_loss.grad(latent_trace)
    for ii in xrange(runs):
        for batch in batcher:
            loss = kyk_loss.value
            if ii % 100 == 0:
                print ii, loss, np.sum(np.power(predicted.value - I_true, 2)) / T
            grad = kyk_loss.grad(latent_trace) + .5 * grad
            latent_trace.value -= learn * grad

    return sigmoid.value



# Sample data from either a GP model or a squid compartment
# t, z, x, inpt, st_axs = sample_gp_model()
t, z, x, inpt, st_axs = sample_squid_model()

with open('squid_' + str(seed) + '_ground.pkl', 'w') as f:
    cPickle.dump((t, z, x, inpt), f)

# raw_input("Press enter to being sampling...\n")
# sample_z_given_x(t, x, inpt, z0=z, axs=st_axs)
z_smpls, gp1_smpls, gp2_smpls = sample_z_given_x(t, x, inpt, N_samples=1000, axs=st_axs, initialize='optimize')
# sample_z_given_x(t, x, inpt, axs=st_axs, z0=z, initialize='ground_truth')
# sample_z_given_x(t, x, inpt, axs=st_axs, initialize='optimize')

with open('squid_' + str(seed) + '_results.pkl', 'w') as f:
    cPickle.dump((z_smpls, gp1_smpls, gp2_smpls), f)


