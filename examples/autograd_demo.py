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
import matplotlib.pyplot as plt
from autograd import grad

def simple_dynamics(z):
    dzdt = 0.01 * np.ones_like(z)
    return dzdt

def forward_euler(z0, T, D):
    z = np.zeros((T,D))

    z[0,:] = z0    
    for t in xrange(1,T):
        z[t,:] = z[t-1,:] + simple_dynamics(z[t-1,:])
    
    return z

def l2_loss(z0, x):
    T,D = x.shape
    z = forward_euler(z0, T, D)
    loss = np.sum((x-z)**2)
    return loss

def generate_test_data(T=100, z0=np.ones(1)):
    z_true = forward_euler(z0, T, 1)

    sigma = 0.1
    x_true = z_true + sigma * np.random.randn(*z_true.shape)

    return z_true, x_true
    
def test_autograd():
    z_true, x_true = generate_test_data()
    
    # Compute the gradient at a range of initial conditions
    z0s = np.linspace(-3,3)[:,None]
    
    loss = lambda z0: l2_loss(z0, x_true)
    dloss_dz0 = grad(loss)

    dz0s = map(dloss_dz0, z0s)

    plt.figure()
    plt.plot(z0s, dz0s)
    plt.show()

test_autograd()
