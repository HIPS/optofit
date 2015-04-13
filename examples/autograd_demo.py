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

def simple_dynamics(z, t):
    #dzdt = 0.01 * np.ones_like(z)
    dzdt = 0.01 * np.ones_like(z) * np.sin(t)
    return dzdt

def hh_dynamics(z, t):
    # Compute dzdt for hodgkin huxley like model
    # z[0]:  V
    # z[1]:  m
    # z[2]:  h
    # z[3]:  n
    dzdt = np.zeros(4)

    # Compute the voltage dynamics
    dzdt += voltage_dynamics(z, t)

    # Compute the channel dynamics
    dzdt += sodium_dynamics(z, t)
    dzdt += potassium_dynamics(z, t)

    # print dzdt
    # import pdb; pdb.set_trace()

    return dzdt

def voltage_dynamics(z, t):
    dzdt = np.zeros(4)
    V = z[0]
    m = z[1]
    h = z[2]
    n = z[3]

    g_leak = 0.3
    g_na = 120
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
    # dVdt += 1.0/self.C * inpt[t,self.i_offset]
    dzdt[0] = dVdt

    return dzdt

def sodium_dynamics(z, t):
    dzdt = np.zeros(4)

    # Use resting potential of zero
    V = z[0] + 60
    m = z[1]
    h = z[2]

    # Compute the alpha and beta as a function of V
    am1 = 0.32*(13.1-V)/(np.exp((13.1-V)/4)-1)
    ah1 = 0.128 * np.exp((17.0-V)/18.0)

    bm1 = 0.28*(V-40.1)/(np.exp((V-40.1)/5.0)-1.0)
    bh1 = 4.0/(1.0 + np.exp((40.-V)/5.0))

    # Compute the channel state updates
    dzdt[1] = am1*(1.-m) - bm1*m
    dzdt[2] = ah1*(1.-h) - bh1*h

    return dzdt

def potassium_dynamics(z, t):
    dzdt = np.zeros(4)

    # Use resting potential of zero
    V = z[0] + 60
    n = z[3]

    # Compute the alpha and beta as a function of V
    an1 = 0.01*(V+55.) /(1-np.exp(-(V+55.)/10.))
    bn1 = 0.125 * np.exp(-(V+65.)/80.)

    # Compute the channel state updates
    dzdt[3] = an1 * (1.0-n) - bn1*n

    return dzdt

#############################################################
# Simulation
#############################################################
dynamics = hh_dynamics

def forward_euler(z0, T, D, dt=1.0):
    z = np.zeros((T,D))

    z[0,:] = z0    
    for t in xrange(1,T):
        z[t,:] = z[t-1,:] + dt*dynamics(z[t-1,:], t)
    
    return z

def l2_loss(z0, x):
    T,D = x.shape
    z = forward_euler(z0, T, D)
    loss = np.sum((x-z)**2)
    return loss

def simulate_and_compute_loss(z0, x, dt=1.0):
    """
    The previous l2_loss function doesn't work because forward euler 
    uses assignments inside its loop. Autograd doesn't take gradients 
    through these assignments and instead silently fails. Here we 
    compute the loss in an online fashion.
    """
    T,D = x.shape
    z = z0
    loss = 0
    for t in xrange(0,T):
        loss += np.sum((x[t,:] - z)**2)
        z += dt * dynamics(z, t)

    return loss

def generate_test_data(T=100, z0=np.ones(1), dt=1.0):
    z_true = forward_euler(z0, T, z0.size, dt=dt)

    sigma = 2.0
    x_true = z_true + sigma * np.random.randn(*z_true.shape)

    return z_true, x_true
    
def simple_test_autograd():
    z0 = -1*np.ones(1)
    z_true, x_true = generate_test_data(z0=z0, dt=1.0)
    plt.plot(z_true)

    # Compute the gradient at a range of initial conditions
    z0s = np.linspace(-3,3)[:,None]

    #loss = lambda z0: l2_loss(z0, x_true)
    loss = lambda z0: simulate_and_compute_loss(z0, x_true)
    dloss_dz0 = ag.grad(loss)

    dz0s = np.array(map(dloss_dz0, z0s))

    plt.figure()
    plt.plot(z0s, dz0s)
    plt.plot(z0s, np.zeros_like(z0s))
    plt.plot(z0, 0, 'ko')
    plt.show()

def hh_test_autograd():
    dt = 0.01
    T = 10.
    t = np.arange(0,T,step=dt)
    z0 = np.zeros(4)
    z0[0] = -40
    z0[1:] = 0.5
    z_true, x_true = generate_test_data(T=len(t), z0=z0, dt=dt)

    # Plot the data
    # plt.plot(t, z_true[:,0], 'b')
    # plt.plot(t, x_true[:,0], 'r')
    # plt.show()

    # Compute the gradient at a range of initial conditions
    V0s = np.linspace(-30,-60, 10)
    m0 = h0 = n0 = 0.5

    #loss = lambda z0: l2_loss(z0, x_true)
    loss = lambda z0: simulate_and_compute_loss(z0, x_true)
    dloss_dz0 = ag.grad(loss)

    dz0s = np.zeros((10, 4))
    for i,V0 in enumerate(V0s):
        z0i = np.array([V0, m0, h0, n0])
        dz0s[i,:] = dloss_dz0(z0i)

    plt.figure()
    plt.plot(V0s, dz0s[:,0])
    plt.plot(V0s, np.zeros_like(V0s))
    plt.plot(z0[0], 0, 'ko')
    plt.show()

hh_test_autograd()
