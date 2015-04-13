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
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import autograd as ag

from hh_dynamics import *

#############################################################
# Simulation
#############################################################
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