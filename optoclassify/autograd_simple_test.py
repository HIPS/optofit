"""
Try out automatic gradients for a simple dynamical system

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

def dynamics(z, t):
    #dzdt = 0.01 * np.ones_like(z)
    dzdt = 0.01 * np.ones_like(z) * np.sin(t)
    return dzdt


#############################################################
# Simulation
#############################################################
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

simple_test_autograd()
