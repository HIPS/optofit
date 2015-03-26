
import os
import copy
import cPickle
import itertools

import numpy as np
seed = np.random.randint(2**16)
# seed = 2958
seed = 58187
#seed = 60017
print "Seed: ", seed

import matplotlib.pyplot as plt
from matplotlib.patches import Path, PathPatch

from hips.inference.particle_mcmc import *
from optofit.cneuron.compartment import SquidCompartment
from optofit.cinference.pmcmc import *

from hips.plotting.layout import  *
import brewer2mpl

colors = brewer2mpl.get_map('Set1', 'Qualitative', 9).mpl_colors

logistic = lambda x: 1.0/(1+np.exp(-x))
logit = lambda p: np.log(p/(1-p))


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


    return td, zd, xd, inptd

def sausage_plot(ax, t, z_mean, z_std, lw=1, alpha=0.5, color='r'):
    """
    Make a sausage plot
    :param ax:
    :param t:
    :param z_mean:
    :param z_std:
    :return:
    """
    T = len(t)
    z_env = np.zeros((T*2,2))
    z_env[:,0] = np.concatenate((t, t[::-1]))
    z_env[:,1] = np.concatenate((z_mean + z_std, z_mean[::-1] - z_std[::-1]))

    ax.add_patch(PathPatch(Path(z_env),
                                facecolor=color,
                                alpha=alpha,
                                edgecolor='none',
                                linewidth=0))

    ax.plot(t, z_mean, color=color, lw=lw)

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
    offset = 6
    z_mean = z_smpls[offset:,...].mean(0)
    z_std = z_smpls[offset:,...].std(0)
    V_inf_mean  = z_smpls[offset:,:,0].mean(0)
    V_inf_std   = z_smpls[offset:,:,0].std(0)
    na_inf_mean = logistic(z_smpls[offset:,:,1]).mean(0)
    na_inf_std  = logistic(z_smpls[offset:,:,1]).std(0)
    k_inf_mean  = logistic(z_smpls[offset:,:,3]).mean(0)
    k_inf_std   = logistic(z_smpls[offset:,:,3]).std(0)

    # Make the figure
    fig = create_figure((6.5,3))

    # Plot the true and inferred voltage
    V_ax = create_axis_at_location(fig, 0.75, 2.375, 5.25, 0.5,
                                   transparent=True, box=False)
    V_ax.plot(t, V_true, 'k', lw=2)
    sausage_plot(V_ax, t, V_inf_mean, V_inf_std, color=colors[0])
    V_ax.set_ylabel('$V \mathrm{ [mV]}$')

    # Plot the true and inferred sodium channel state
    na_ax = create_axis_at_location(fig, 0.75, 1.625, 5.25, 0.5,
                                   transparent=True, box=False)
    na_ax.plot(t, na_true, 'k', lw=2)
    sausage_plot(na_ax, t, na_inf_mean, na_inf_std, color=colors[0])
    na_ax.set_ylabel('$\sigma(z_{Na})$')
    na_ax.set_ylim([0,0.3])

    # Plot the true and inferred sodium channel state
    k_ax = create_axis_at_location(fig, 0.75, .875, 5.25, 0.5,
                                   transparent=True, box=False)
    k_ax.plot(t, k_true, 'k', lw=2)
    sausage_plot(k_ax, t, k_inf_mean, k_inf_std, color=colors[0])
    k_ax.set_ylabel('$\sigma(z_{K})$')
    k_ax.set_ylim([0,1])

    # Plot the driving current
    I_ax = create_axis_at_location(fig, 0.75, 0.375, 5.25, 0.25,
                                   transparent=True, box=False)
    I_ax.plot(t, inpt, 'k', lw=2)
    I_ax.set_ylabel('$I \mathrm{}$')
    I_ax.set_yticks([0,4,8])
    I_ax.set_ylim([-2,10])
    I_ax.set_xlabel('$\mathrm{time [ms]}$')

    plt.savefig(os.path.join('cosyne', 'figure1.pdf'))
    plt.ioff()
    plt.show()

def make_figure_2(gpk_smpls):
    grid = 100
    z_min = logit(0.001)
    z_max = logit(0.999)
    V_min = -80.
    V_max = 50.
    Z = np.array(list(
              itertools.product(*([np.linspace(z_min, z_max, grid) for _ in range(1)]
                                + [np.linspace(V_min, V_max, grid)]))))

    h_smpls = []
    for gps in gpk_smpls:
        m_pred, _, _, _ = gps[0].predict(Z)
        h_smpls.append(m_pred)

    h_mean = np.array(h_smpls).mean(0)
    h_mean = h_mean.reshape((grid, grid))

    fig = create_figure((2,2))
    ax = create_axis_at_location(fig, .5, .5, 1, 1, box=True, transparent=True)

    print "h_lim: ", np.amin(h_mean), " ", np.amax(h_mean)

    im = ax.imshow(h_mean, extent=(V_min, V_max, z_max, z_min), cmap='RdGy',
                   vmin=-3, vmax=3)
    ax.set_aspect((V_max-V_min)/(z_max-z_min))
    ax.set_ylabel('$z_{K}$')
    ax.set_xlabel('$V$')
    ax.set_title('$\\frac{\mathrm{d}z_{K}}{\mathrm{d}t}(z_{K},V)$')

    ax.set_xticks([-80, -40, 0, 40])
    fig.savefig('dk_dt.pdf')

def make_figure_3():
    z_min = logit(0.001)
    z_max = logit(0.999)
    V_min = -80.
    V_max = 50.
    dlogit = lambda x: 1./(x*(1-x))

    g = lambda x: x**4
    ginv = lambda u: u**(1./4)
    dg_dx = lambda x: 4*x**3

    u_to_x = lambda u: ginv(logistic(u))
    x_to_u = lambda x: logit(g(x))

    uu = np.linspace(-6,0,1000)
    xx = u_to_x(uu)
    #g = lambda x: x
    #ginv = lambda u: u
    #dg_dx = lambda x: 1.0


    # Compute dynamics du/dt
    alpha = lambda V: 0.01 * (10.01-V) / (np.exp((10.01-V)/10.) - 1)
    beta = lambda V: 0.125 * np.exp(-V/80.)
    dx_dt = lambda x,V: alpha(V)*(1-x) - beta(V) * x
    du_dt = lambda u,V: dlogit(g(u_to_x(u))) * dg_dx(u_to_x(u)) * dx_dt(u_to_x(u),V)

    # Plot the change in u as a function of u and V
    V = np.linspace(0,(V_max-V_min),100)

    fig = create_figure((2,2))
    ax = create_axis_at_location(fig, .5, .5, 1, 1, box=True, transparent=True)

    ax.imshow(du_dt(uu[:,None], V[None,:]),
               extent=[V_min, V_max, uu[-1], uu[0]],
               interpolation="none",
               cmap='RdGy')
    ax.set_xlabel('V')
    ax.set_aspect((V_max-V_min)/(z_max-z_min))
    ax.set_ylabel('u')
    ax.set_title('du_dt(u,V)')

    # ax2 = fig.add_subplot(1,2,2)
    # ax2.imshow(dx_dt(xx[:,None], V[None,:]),
    #            extent=[V[0], V[-1], xx[-1], xx[0]],
    #            interpolation="none",
    #            cmap=plt.cm.Reds)
    # ax2.set_aspect(100)
    # ax2.set_xlabel('V')
    # ax2.set_ylabel('x')
    # ax2.set_title('dx_dt(x,V)')
    plt.ioff()
    plt.show()

def make_figure_4():
    logit = lambda x: np.log(x / (1-x))
    logistic = lambda u: np.exp(u) / (1 + np.exp(u))
    dlogit = lambda x: 1./(x*(1-x))

    g = lambda x: x**4
    ginv = lambda u: u**(1./4)
    dg_dx = lambda x: 4*x**3

    u_to_x = lambda u: ginv(logistic(u))
    x_to_u = lambda x: logit(g(x))

    uu = np.linspace(-6,6,1000)
    xx = u_to_x(uu)

    # Compute dynamics du/dt 
    alpha = lambda V: 0.01 * (10.01-V) / (np.exp((10.01-V)/10.) - 1)
    beta = lambda V: 0.125 * np.exp(-V/80.)
    dx_dt = lambda x,V: alpha(V)*(1-x) - beta(V) * x
    du_dt = lambda u,V: dlogit(g(u_to_x(u))) * dg_dx(u_to_x(u)) * dx_dt(u_to_x(u),V)

    # Plot the change in u as a function of u and V
    V = np.linspace(0,100,100)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(du_dt(uu[:,None], V[None,:]), 
               extent=[V[0], V[-1], uu[-1], uu[0]], 
               interpolation="none",
               cmap=plt.cm.Reds)
    ax1.set_aspect(20)
    ax1.set_xlabel('V')
    ax1.set_ylabel('u')
    ax1.set_title('du_dt(u,V)')

    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(dx_dt(xx[:,None], V[None,:]), 
               extent=[V[0], V[-1], xx[-1], xx[0]], 
               interpolation="none",
               cmap=plt.cm.Reds)
    ax2.set_aspect(100)
    ax2.set_xlabel('V')
    ax2.set_ylabel('x')
    ax2.set_title('dx_dt(x,V)')
    plt.show()

def make_figure_5(gpk_smpls):
    g = lambda x: x**4
    ginv = lambda u: u**(1./4)
    dg_dx = lambda x: 4*x**3

    u_to_x = lambda u: ginv(logistic(u))
    x_to_u = lambda x: logit(g(x))
    dlogit = lambda x: 1./(x*(1-x))

    uu = np.linspace(-6,6,100)
    xx = u_to_x(uu)

    # Compute dynamics du/dt 
    alpha = lambda V: 0.01 * (10.01-V) / (np.exp((10.01-V)/10.) - 1)
    beta = lambda V: 0.125 * np.exp(-V/80.)
    dx_dt = lambda x,V: alpha(V)*(1-x) - beta(V) * x
    du_dt = lambda u,V: dlogit(g(u_to_x(u))) * dg_dx(u_to_x(u)) * dx_dt(u_to_x(u),V)

    grid = 100
    z_min = logit(0.001)
    z_max = logit(0.999)
    V_min = -80
    V_max = 50

    zz = np.linspace(z_min, z_max, grid)
    V_gp = np.linspace(V_min, V_max, grid)
    Z = np.array(list(
              itertools.product(*([zz for _ in range(1)]
                                  + [V_gp]))))

    h_smpls = []
    for gps in gpk_smpls:
        m_pred, _, _, _ = gps[0].predict(Z)
        h_smpls.append(m_pred)

    h_mean = np.array(h_smpls).mean(0)
    h_mean = h_mean.reshape((grid, grid))

    # Plot the change in u as a function of u and V
    
    def dsig(z):
        sigz = logistic(z)
        return np.multiply(sigz, 1 - sigz)

    df_dt = lambda z, dzdt: np.multiply(dsig(z), dzdt)
    fig = plt.figure()

    ax1 = fig.add_subplot(2,2,1)
    dudt = du_dt(uu[:,None], V_gp[None,:])
    v_max = max((np.max(dudt), np.max(h_mean)))
    v_min = min((np.min(dudt), np.min(h_mean)))
    ax1.imshow(du_dt(uu[:,None], V_gp[None,:]), 
               extent=[V_gp[0], V_gp[-1], uu[-1], uu[0]], 
               interpolation="none",
               cmap=plt.cm.Reds,
               vmin=v_min,
               vmax=v_max)
    ax1.set_aspect(20)
    ax1.set_xlabel('V')
    ax1.set_ylabel('latent state')
    ax1.set_title('Ground Truth: dz_dt(z,V)')

    ax2 = fig.add_subplot(2,2,3)
    ax2.imshow(h_mean, 
               extent=[V_gp[0], V_gp[-1], uu[-1], uu[0]], 
               interpolation="none",
               cmap=plt.cm.Reds,
               vmin=v_min,
               vmax=v_max)
    ax2.set_aspect(20)
    ax2.set_xlabel('V')
    ax2.set_ylabel('latent state')
    ax2.set_title('Inferred: dz_dt(z,V)')
    
    ax1 = fig.add_subplot(2,2,2)
    ax1.imshow(uu[:, None] * dg_dx(u_to_x(uu[:, None])) * dx_dt(u_to_x(uu[:, None]), V_gp[None, :]+60),
               extent=[V_gp[0], V_gp[-1], xx[-1], xx[0]],
               interpolation="none",
               cmap=plt.cm.Reds,
               vmin=-1,
               vmax=.5)
    ax1.set_aspect(100)
    ax1.set_xlabel('V')
    ax1.set_ylabel('open fraction')
    ax1.set_title('Ground Truth: df_dt(f,V)')

    ax2 = fig.add_subplot(2,2,4)
    ax2.imshow(df_dt(np.array([zz for a in range(grid)]).transpose(), h_smpls[0].reshape((grid, grid))),
               extent=[V_gp[0], V_gp[-1], xx[-1], xx[0]], 
               interpolation="none",
               cmap=plt.cm.Reds,
               vmin=-1,
               vmax=.5)
    ax2.set_aspect(100)
    ax2.set_xlabel('V')
    ax2.set_ylabel('open fraction')
    ax2.set_title('Inferred: df_dt(f,V)')
    plt.show()


    
    def plot_at_x(ax, index):
        mean = uu[:, None] * dg_dx(u_to_x(uu[:, None])) * dx_dt(u_to_x(uu[:, None]), V_gp[None, :]+60)
        mean = mean[index, :]
        #std = 0.0001 * np.ones(mean.shape)
        voltage = V_gp

        color = 'r'
        ax.plot(voltage, mean, color=color)
        #ax.fill_between(voltage, mean - std, mean + std, color=color, alpha = 0.5)

        mean, _, dzdt_low, dzdt_high = gpk_smpls[7][0].predict(Z) #62
        mean      = mean.reshape((grid, grid))
        dzdt_low  = dzdt_low.reshape((grid, grid))
        dzdt_high = dzdt_high.reshape((grid, grid))
        
        zs = np.array([zz for b in range(grid)]).transpose()
        dfdt_mean = df_dt(zs, mean)
        dfdt_low  = df_dt(zs, dzdt_low)
        dfdt_high = df_dt(zs, dzdt_high)

        color = 'b'
        ax.plot(voltage, dfdt_mean[index, :], color=color)
        ax.fill_between(voltage, dfdt_low[index, :], dfdt_high[index, :], color=color, alpha = 0.5)

    f, axs = plt.subplots(9, sharex=True)
    for i in range(len(axs)):
        plot_at_x(axs[i], i*2 + 42)
    plt.show()

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.imshow((uu[:, None] * dg_dx(u_to_x(uu[:, None])) * dx_dt(u_to_x(uu[:, None]), V_gp[None, :]+60)) - h_mean,
            extent=[V_gp[0], V_gp[-1], xx[-1], xx[0]],
            cmap=plt.cm.RdGy,
            vmin=-.5,
            vmax=.5,
    )
    ax.set_aspect(100)
    plt.show()

def make_figure_7(z_smpls, gpk_smpls):
    g = lambda x: x**4
    ginv = lambda u: u**(1./4)
    dg_dx = lambda x: 4*x**3

    u_to_x = lambda u: ginv(logistic(u))
    x_to_u = lambda x: logit(g(x))
    dlogit = lambda x: 1./(x*(1-x))

    uu = np.linspace(-6,6,100)
    xx = u_to_x(uu)

    # Compute dynamics du/dt 
    alpha = lambda V: 0.01 * (10.01-V) / (np.exp((10.01-V)/10.) - 1)
    beta = lambda V: 0.125 * np.exp(-V/80.)
    dx_dt = lambda x,V: alpha(V)*(1-x) - beta(V) * x
    du_dt = lambda u,V: dlogit(g(u_to_x(u))) * dg_dx(u_to_x(u)) * dx_dt(u_to_x(u),V)

    grid = 100
    z_min = logit(0.001)
    z_max = logit(0.999)
    V_min = -80
    V_max = 50

    zz = np.linspace(z_min, z_max, grid)
    V_gp = np.linspace(V_min, V_max, grid)
    Z = np.array(list(
              itertools.product(*([zz for _ in range(1)]
                                  + [V_gp]))))

    h_smpls = []
    for gps in gpk_smpls:
        m_pred, _, _, _ = gps[0].predict(Z)
        h_smpls.append(m_pred)

    h_mean = np.array(h_smpls).mean(0)
    h_mean = h_mean.reshape((grid, grid))

    # Plot the change in u as a function of u and V
    
    def dsig(z):
        sigz = logistic(z)
        return np.multiply(sigz, 1 - sigz)

    df_dt = lambda z, dzdt: np.multiply(dsig(z), dzdt)

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.imshow((uu[:, None] * dg_dx(u_to_x(uu[:, None])) * dx_dt(u_to_x(uu[:, None]), V_gp[None, :]+60)) - df_dt(np.array([zz for a in range(grid)]).transpose(), h_mean),
            extent=[V_gp[0], V_gp[-1], xx[-1], xx[0]],
            cmap=plt.cm.RdGy
    )
    ax.set_aspect(100)
    ax.scatter(z_smpls[:11, :, 0].reshape((11*3000)), logistic(z_smpls[:11, :, 3].reshape((11*3000))))
    ax.set_title("Errors")
    plt.show()
    
# Simulate the squid compartment to get the ground truth
t, z_true, x, inpt = sample_squid_model()

# Load the results of the pMCMC inference
with open('squid2_results5.pkl', 'r') as f:
    z_smpls, gpna_smpls, gpk_smpls = cPickle.load(f)

burn = 30
z_smpls    = z_smpls[burn:]
gpna_smpls = gpna_smpls[burn:]
gpk_smpls  = gpk_smpls[burn:]

make_figure_1(t, inpt, z_true, z_smpls, gpna_smpls, gpk_smpls)
#make_figure_2(gpk_smpls)
#make_figure_3()
#make_figure_4()
make_figure_5(gpk_smpls)
make_figure_7(z_smpls, gpk_smpls)
