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
np.seterr(all='raise')
import matplotlib.pyplot as plt
import autograd as ag

# def hh_dynamics(z, t):
#     # Compute dzdt for hodgkin huxley like model
#     # z[0]:  V
#     # z[1]:  m
#     # z[2]:  h
#     # z[3]:  n
#     dzdt = np.zeros(4)
#
#     # Compute the voltage dynamics
#     dzdt += voltage_dynamics(z, t)
#
#     # Compute the channel dynamics
#     dzdt += sodium_dynamics(z, t)
#     dzdt += potassium_dynamics(z, t)
#
#     # print dzdt
#     # import pdb; pdb.set_trace()
#
#     return dzdt

def voltage_dynamics(V, m, h, n, t, g_na):
    g_leak = 0.3
    # g_na = 120
    g_k = 36

    # Compute the voltage drop through each channel
    V_leak = V - (-60.)
    V_na = m**3 * h * (V-50.)
    V_k = n**4 * (V-(-77.))

    I_ionic = g_leak * V_leak
    I_ionic += g_na * V_na
    I_ionic += g_k * V_k

    # dVdt[t,n] = -1.0/self.C * I_ionic
    dVdt = -1.0 * I_ionic

    # Add in driving current
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
def forward_euler(z0, T, dt=1.0, g_na=120.):
    z = np.zeros((T,4))

    z[0,:] = z0    
    for t in xrange(1,T):
        # Compute dynamics
        Vtm1 = z[t-1,0]
        mtm1 = z[t-1,1]
        htm1 = z[t-1,2]
        ntm1 = z[t-1,3]

        dvdt = voltage_dynamics(Vtm1, mtm1, htm1, ntm1, t, g_na)
        dmdt, dhdt = sodium_dynamics(Vtm1, mtm1, htm1, t)
        dndt = potassium_dynamics(Vtm1, ntm1, t)

        # Update state
        z[t,0] = z[t-1,0] + dt*dvdt
        z[t,1] = z[t-1,1] + dt*dmdt
        z[t,2] = z[t-1,2] + dt*dhdt
        z[t,3] = z[t-1,3] + dt*dndt

    return z

def simulate_and_compute_loss(v0, g_na, x, dt=1.0):
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
        dvdt = voltage_dynamics(v, m, h, n, t, g_na)
        dmdt, dhdt = sodium_dynamics(v, m, h, t)
        dndt = potassium_dynamics(v, n, t)

        v += dt * dvdt
        m += dt * dmdt
        h += dt * dhdt
        n += dt * dndt

    return loss

def generate_test_data(T=100, z0=np.ones(1), dt=1.0, g_na=120.):
    z_true = forward_euler(z0, T, dt=dt, g_na=g_na)

    sigma = 10.0
    x_true = z_true + sigma * np.random.randn(*z_true.shape)

    return z_true, x_true


def hh_test_autograd():
    dt = 0.01
    T = 2.
    t = np.arange(0,T,step=dt)
    z0 = np.zeros(4)
    z0[0] = -40
    z0[1:] = 0.5
    g_na = 120.
    z_true, x_true = generate_test_data(T=len(t), z0=z0, dt=dt, g_na=g_na)

    # Plot the data
    plt.plot(t, z_true[:,0], 'b')
    plt.plot(t, x_true[:,0], 'r')
    plt.show()

    # # Compute the gradient, dLoss/dV_0 a range of initial conditions
    # V0s = np.linspace(-30, -50, 5)
    # loss = lambda v0: simulate_and_compute_loss(v0, g_na, x_true, dt=dt)
    # dloss_dv0 = ag.grad(loss)
    #
    # dv0s = np.zeros_like(V0s)
    # for i,V0 in enumerate(V0s):
    #     print "Computing gradient for ", V0
    #     dv0s[i] = dloss_dv0(V0)
    #
    # plt.figure()
    # plt.plot(V0s, dv0s)
    # plt.plot(V0s, np.zeros_like(V0s))
    # plt.plot(z0[0], 0, 'ko')
    # plt.show()

    # Compute the gradient, dLoss/dg_na at a range of initial conditions
    gs = np.linspace(100, 150, 5)
    loss = lambda g_na: simulate_and_compute_loss(z0[0], g_na, x_true, dt=dt)
    dloss_dgna = ag.grad(loss)

    dgs = np.zeros_like(gs)
    for i,g in enumerate(gs):
        print "Computing gradient for ", g
        dgs[i] = dloss_dgna(g)

    plt.figure()
    plt.plot(gs, dgs)
    plt.plot(gs, np.zeros_like(gs))
    plt.plot(g_na, 0, 'ko')
    plt.show()

hh_test_autograd()
