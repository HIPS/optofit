"""
Try out automatic gradients for Hodgkin-Huxley dynamics.

The scalar output is the likelihood of the observed membrane
potential under the Hodgkin-Huxley model with parameters
\theta and latent state trajectories Z

In Shamim's work, there is a fixed set of global parameters.
Classification is based on the marginal latent state probabilities
with those fixed global dynamics parameters.

Here we expect the neurons to have different global parameters,
e.g. different channel densities or channel kinetics. In theory,
the classifier would work by first inferring the latent states,
then inferring the channel densities, and finally classifying based
on those channel densities. How can we backpropagate through all of
these steps?

Are these the right parameters to classify based on? Or should
we instead be classifying based on the latent state trajectories
themselves?

The input to the classifier is a time series of neural responses
to a particular stimulus. The output should be a multinomial class
label.

"""

import numpy as np
# np.seterr(all='raise')
import matplotlib.pyplot as plt
import autograd as ag
from scipy.optimize import minimize

# Define a few constants
E_leak = -60.       # Reversal potentials
E_na   = 50.
E_k    = -77.
g_leak = 0.3        # Conductances
g_na   = 120
g_k    = 36


#############################################################
# Dynamics functions
#############################################################
def voltage_dynamics(V, m, h, n, t, g, inpt):

    # Compute the voltage drop through each channel
    V_leak = V - E_leak
    V_na = m**3 * h * (V-E_na)
    V_k = n**4 * (V-E_k)

    I_ionic = g[0] * V_leak
    I_ionic += g[1] * V_na
    I_ionic += g[2] * V_k

    # dVdt[t,n] = -1.0/self.C * I_ionic
    dVdt = -1.0 * I_ionic

    # Add in driving current
    dVdt += inpt[t]

    return dVdt

def sodium_dynamics(V, m, h, t):

    # Use resting potential of zero
    V_ref = V + 60

    # Compute the alpha and beta as a function of V
    am1 = 0.32*(13.1-V_ref)/(np.exp((13.1-V_ref)/4)-1)
    ah1 = 0.128 * np.exp((17.0-V_ref)/18.0)

    bm1 = 0.28*(V_ref-40.1)/(np.exp((V_ref-40.1)/5.0)-1.0)
    bh1 = 4.0/(1.0 + np.exp((40.-V_ref)/5.0))

    dmdt = am1*(1.-m) - bm1*m
    dhdt = ah1*(1.-h) - bh1*h

    return dmdt, dhdt

def potassium_dynamics(V, n, t):
    # Use resting potential of zero
    V_ref = V+60

    # Compute the alpha and beta as a function of V
    an1 = 0.01*(V_ref+55.) /(1-np.exp(-(V_ref+55.)/10.))
    bn1 = 0.125 * np.exp(-(V_ref+65.)/80.)

    # Compute the channel state updates
    dndt = an1 * (1.0-n) - bn1*n

    return dndt

#############################################################
# Simulation
#############################################################
def forward_euler(z0, g, inpt, T, dt=1.0):
    z = np.zeros((T,4))

    z[0,:] = z0    
    for t in xrange(1,T):
        # Compute dynamics
        Vtm1 = z[t-1,0]
        mtm1 = z[t-1,1]
        htm1 = z[t-1,2]
        ntm1 = z[t-1,3]

        dvdt = voltage_dynamics(Vtm1, mtm1, htm1, ntm1, t, g, inpt)
        dmdt, dhdt = sodium_dynamics(Vtm1, mtm1, htm1, t)
        dndt = potassium_dynamics(Vtm1, ntm1, t)

        # Update state
        z[t,0] = z[t-1,0] + dt*dvdt
        z[t,1] = z[t-1,1] + dt*dmdt
        z[t,2] = z[t-1,2] + dt*dhdt
        z[t,3] = z[t-1,3] + dt*dndt

    return z

def simulate_and_compute_loss(v0, g, inpt, x, dt=1.0):
    """
    Autograd doesn't take gradients through assignments.
    Here we compute the loss in an online fashion.
    """
    T,D = x.shape
    v = v0
    m = 0.5
    h = 0.5
    n = 0.5

    loss = 0
    for t in xrange(0,T):
        loss += np.sum((x[t,0] - v)**2)

        # Compute dynamics
        dvdt = voltage_dynamics(v, m, h, n, t, g, inpt)
        dmdt, dhdt = sodium_dynamics(v, m, h, t)
        dndt = potassium_dynamics(v, n, t)

        v += dt * dvdt
        m += dt * dmdt
        h += dt * dhdt
        n += dt * dndt

    return loss / T

#############################################################
# Optimization
#############################################################
def optimize_conductances():
    """
    Simple demo to optimize the conductance values for a given set of
    neural dynamics
    :return:
    """
    # Timing
    dt = 0.01
    T = 2000     # number of time steps
    t = np.arange(T) * dt

    # Define an input
    t_on  = 5.
    t_off = 15.
    i_amp = 1000.
    inpt  = i_amp * (t > t_on)  * (t < t_off)

    # True initial conditions
    z0_true = np.zeros(4)
    v0_true = -40
    z0_true[0] = v0_true
    z0_true[1:] = 0.5

    # True conductance values
    g_true = np.array([0.2, 120., 36.])

    # Simulate true data
    z_true = forward_euler(z0_true, g_true, inpt, T, dt=dt)
    sigma = 3.0
    x_true = z_true + sigma * np.random.randn(*z_true.shape)

    # Initialize values to be optimized
    g_inf = 10 * np.ones(3)
    z_inf = forward_euler(z0_true, g_inf, inpt, T, dt=dt)

    # Plot the voltage traces
    plt.ion()
    plt.figure()
    plt.subplot(121)
    plt.plot(t, z_true[:,0], 'k', lw=2)
    plt.plot(t, x_true[:,0], 'r')
    ln = plt.plot(t, z_inf[:,0], 'b', lw=2)[0]
    plt.xlabel("time [ms]")
    plt.ylabel("V [mV]")

    # Plot the conductances
    plt.subplot(122)
    plt.bar(np.arange(3), g_true, color='k', alpha=0.5)
    bars = plt.bar(np.arange(3), g_inf, color='r', alpha=0.5)
    plt.xticks(0.5 + np.arange(3), ["leak", "Na", "Kdr"])
    plt.xlabel("Channel")
    plt.ylabel("Conductance [mS]")
    plt.pause(0.001)

    # Compute a gradient function as a function of g
    loss = lambda g: simulate_and_compute_loss(v0_true, g, inpt, x_true, dt=dt)
    dloss_dg = ag.grad(loss)

    # Make a callback to plot
    itr = [0]
    def callback(g_inf):
        print "Iteration ", itr[0]
        itr[0] = itr[0] + 1
        z_inf = forward_euler(z0_true, g_inf, inpt, T, dt=dt)
        ln.set_data(t, z_inf[:,0])
        for i,bar in enumerate(bars):
            bar.set_height(g_inf[i])
        plt.pause(0.001)

    # Optimize with scipy
    bnds = [(0, None)] * 3
    g_inf = minimize(loss, g_inf, jac=dloss_dg, bounds=bnds, callback=callback).x

    print "True g:"
    print g_true
    print ""
    print "Inferred g: "
    print g_inf

optimize_conductances()