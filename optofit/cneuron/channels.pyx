# distutils: extra_compile_args = -O3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True

from cython.parallel cimport prange

from component cimport Component
from compartment cimport Compartment

import numpy as np
cimport numpy as np

import GPy

cdef class Channel(Component):
    """
    Abstract base class for an ion channel.
    """

    def __init__(self, name=None):
        super(Channel, self).__init__(name=name)

        # All channels (at least so far!) have a conductance and a reversal
        # potential

    cpdef double current(self, double[:,:,::1] x, double V, int t, int n):
        pass

cdef class LeakChannel(Channel):
    """
    Passive leak channel.
    """
    def __init__(self, name=None, hypers=None):
        super(LeakChannel, self).__init__(name=name)

        # Set the hyperparameters
        # TODO: Use distributions to set the hypers?
        self.g = hypers['g_leak']
        self.E = hypers['E_leak']

    cpdef double current(self, double[:,:,::1] x, double V, int t, int n):
        """
        Evaluate the instantaneous current through this channel
        """
        return V - self.E

cdef class NaChannel(Channel):
    """
    Sodium channel.
    """
    def __init__(self, name=None, hypers=None):
        super(NaChannel, self).__init__(name=name)

        # Specify the number of parameters
        self.n_x = 2

        # Set the hyperparameters
        # TODO: Use distributions to set the hypers?
        self.g = hypers['g_na']
        self.E = hypers['E_na']

    def steady_state(self, x0):
        """
        Set the steady state

        x0:    a buffer into which the steady state should be placed
        """
        V = x0[self.parent_compartment.x_offset]
        # Steady state value of the latent vars
        # Compute the alpha and beta as a function of V
        am1 = 0.1*(V+35.)/(1-exp(-(V+35.)/10.))
        ah1 = 0.07*exp(-(V+50.)/20.)
        bm1 = 4.*exp(-(V+65.)/18.)
        bh1 = 1./(exp(-(V+35)/10.)+1)

        x0[self.x_offset+0] = am1/(am1+bm1)
        x0[self.x_offset+1] = ah1/(ah1+bh1)

    cpdef double current(self, double[:,:,::1] x, double V, int t, int n):
        """
        Evaluate the instantaneous current through this channel
        """
        cdef double m = x[t,n,self.x_offset+0]
        cdef double h = x[t,n,self.x_offset+1]
        return m**3 * h * (V - self.E)

    cpdef kinetics(self, double[:,:,::1] dxdt, double[:,:,::1] x, double[:,::1] inpt, int[::1] ts):
        cdef int T = x.shape[0]
        cdef int N = x.shape[1]
        cdef int D = x.shape[2]
        cdef int M = inpt.shape[1]
        cdef int S = ts.shape[0]
        cdef int n, s, t
        cdef double am1, ah1, bm1, bh1

        # Get a pointer to the voltage of the parent compartment
        # TODO: This approach sucks b/c it assumes the voltage
        # is the first compartment state. It should be faster than
        # calling back into the parent to have it extract the voltage
        # for us though.
        # cdef double[:,:] V = x[:,:,self.parent_compartment.x_offset]
        # cdef double[:,:] m = x[:,:,self.x_offset+0]
        # cdef double[:,:] h = x[:,:,self.x_offset+1]
        cdef double V, m, h

        with nogil:
            for s in prange(S):
                t = ts[s]
                for n in prange(N):
                    # # Compute the alpha and beta as a function of V
                    # am1 = 0.1*(V[t,n]+35.)/(1-exp(-(V[t,n]+35.)/10.))
                    # ah1 = 0.07*exp(-(V[t,n]+50.)/20.)
                    #
                    # bm1 = 4.*exp(-(V[t,n]+65.)/18.)
                    # bh1 = 1./(exp(-(V[t,n]+35.)/10.)+1.)
                    #
                    # # Compute the channel state updates
                    # dxdt[t,n,self.x_offset+0] = am1*(1.-m[t,n]) - bm1*m[t,n]
                    # dxdt[t,n,self.x_offset+1] = ah1*(1.-h[t,n]) - bh1*h[t,n]

                    # Compute the alpha and beta as a function of V
                    V = x[t,n,self.parent_compartment.x_offset]
                    m = x[t,n,self.x_offset]
                    h = x[t,n,self.x_offset+1]
                    am1 = 0.1*(V+35.)/(1-exp(-(V+35.)/10.))
                    ah1 = 0.07*exp(-(V+50.)/20.)

                    bm1 = 4.*exp(-(V+65.)/18.)
                    bh1 = 1./(exp(-(V+35.)/10.)+1.)

                    # Compute the channel state updates
                    dxdt[t,n,self.x_offset+0] = am1*(1.-m) - bm1*m
                    dxdt[t,n,self.x_offset+1] = ah1*(1.-h) - bh1*h


cdef class KdrChannel(Channel):
    """
    Potassium (delayed rectification) channel.
    """
    def __init__(self, name=None, hypers=None):
        super(KdrChannel, self).__init__(name=name)

        # Specify the number of parameters
        self.n_x = 1

        # Set the hyperparameters
        # TODO: Use distributions to set the hypers?
        self.g = hypers['g_kdr']
        self.E = hypers['E_kdr']

    def steady_state(self, x0):
        V = x0[self.parent_compartment.x_offset]
        # Steady state value of the latent vars
        an1 = 0.01*(V+55.)/(1-exp(-(V+55.)/10.))
        bn1 = 0.125*exp(-(V+65.)/80.)

        x0[self.x_offset] = an1/(an1+bn1)

    cpdef double current(self, double[:,:,::1] x, double V, int t, int n):
        """
        Evaluate the instantaneous current through this channel
        """
        cdef double nn = x[t,n,self.x_offset]
        return nn**4 * (V - self.E)

    cpdef kinetics(self, double[:,:,::1] dxdt, double[:,:,::1] x, double[:,::1] inpt, int[::1] ts):
        cdef int T = x.shape[0]
        cdef int N = x.shape[1]
        cdef int D = x.shape[2]
        cdef int M = inpt.shape[1]
        cdef int S = ts.shape[0]
        cdef int n, s, t
        cdef double an1, bn1

        # Get a pointer to the voltage of the parent compartment
        # TODO: This approach sucks b/c it assumes the voltage
        # is the first compartment state. It should be faster than
        # calling back into the parent to have it extract the voltage
        # for us though.
        # cdef double[:,:] V = x[:,:,self.parent_compartment.x_offset]
        # cdef double[:,:] nn = x[:,:,self.x_offset+0]

        cdef double V, nn

        with nogil:
            for s in prange(S):
                t = ts[s]
                for n in prange(N):
                    V = x[t,n,self.parent_compartment.x_offset]
                    nn = x[t,n,self.x_offset]

                    # Compute the alpha and beta as a function of V
                    an1 = 0.01*(V+55.) /(1-exp(-(V+55.)/10.))
                    bn1 = 0.125*exp(-(V+65.)/80.)

                    # Compute the channel state updates
                    dxdt[t,n,self.x_offset] = an1*(1.-nn) - bn1*nn


        return dxdt


cdef class GPChannel(Channel):
    """
    Generic channel with a Gaussian process dyamics function
    """
    def __init__(self, name=None, hypers=None):
        super(GPChannel, self).__init__(name=name)

        # Specify the number of parameters
        self.n_x = 1

        # Set the hyperparameters
        self.g = hypers['g_kdr']
        self.E = hypers['E_kdr']

        # Import stuff
        import GPy
        import itertools
        import numpy as np

        # Create a GP Object for the dynamics model
        # This is a function Z x V -> dZ, i.e. a 2D
        # Gaussian process regression.
        # Lay out a grid of inducing points for a sparse GP

        print self.g
        self.grid = 10
        #self.z_min = sigma_inv(0.005)
        #self.z_max = sigma_inv(0.995)
        self.z_min = -6.0
        self.z_max = 6.0
        self.V_min = -65.
        self.V_max = 120.
        self.Z = np.array(list(
                      itertools.product(*([np.linspace(self.z_min, self.z_max, self.grid) for _ in range(self.n_x)]
                                        + [np.linspace(self.V_min, self.V_max, self.grid)]))))

        # Create independent RBF kernels over Z and V
        kernel_z_hyps = { 'input_dim' : 1,
                          'variance' : 1,
                          'lengthscale' : (self.z_max-self.z_min)/4.}

        self.kernel_z = GPy.kern.rbf(**kernel_z_hyps)
        for n in range(1, self.n_x):
            self.kernel_z = self.kernel_z.prod(GPy.kern.rbf(kernel_z_hyps))
        print self.kernel_z

        kernel_V_hyps = { 'input_dim' : 1,
                          'variance' : 1,
                          'lengthscale' : (self.V_max-self.V_min)/4.}
        self.kernel_V = GPy.kern.rbf(**kernel_V_hyps)
        print self.kernel_V

        # Combine the kernel for z and V
        self.kernel = self.kernel_z.prod(self.kernel_V, tensor=True)
        print self.kernel_V

        # Initialize with a random sample from the prior
        m = np.zeros(self.Z.shape[0])
        C = self.kernel.K(self.Z)
        print m
        print C
        self.h = np.random.multivariate_normal(m, C, 1).T

        # # Instatiate a sparse GP regression model
        # self.gp = GPy.models.SparseGPRegression(np.zeros((1, self.n_x+1)),
        #                                         np.zeros((1,1)),
        #                                         self.kernel,
        #                                         Z=self.Z)
        #
        # # HACK: Rather than using a truly nonparametric approach, just sample
        # # the GP at the grid of inducing points and interpolate at the GP mean
        # self.h = self.gp.posterior_samples(self.Z, size=1)

        # Create a sparse GP model with the sampled function h
        # This will be used for prediction
        self.gp = GPy.models.SparseGPRegression(self.Z, self.h, self.kernel, Z=self.Z)


    def steady_state(self, x0):
        V = x0[self.parent_compartment.x_offset]

        # TODO: Set the steady state
        x0[self.x_offset] = 0

    cpdef double current(self, double[:,:,::1] x, double V, int t, int n):
        """
        Evaluate the instantaneous current through this channel
        """
        cdef double z = x[t,n,self.x_offset]
        return sigma(z) * (V - self.E)

    cpdef kinetics(self, double[:,:,::1] dxdt, double[:,:,::1] x, double[:,::1] inpt, int[::1] ts):
        cdef int T = x.shape[0]
        cdef int N = x.shape[1]
        cdef int D = x.shape[2]
        cdef int M = inpt.shape[1]
        cdef int S = ts.shape[0]
        cdef int n, s, t
        cdef double an1, bn1

        # Get a pointer to the voltage of the parent compartment
        # TODO: This approach sucks b/c it assumes the voltage
        # is the first compartment state. It should be faster than
        # calling back into the parent to have it extract the voltage
        # for us though.
        # cdef double[:,:] V = x[:,:,self.parent_compartment.x_offset]
        # cdef double[:,:] nn = x[:,:,self.x_offset+0]

        cdef double V, z

        for s in range(S):
            t = ts[s]
            for n in range(N):
                V = x[t,n,self.parent_compartment.x_offset]
                z = x[t,n,self.x_offset]

                # Sample from the GP kinetics model
                m_pred, v_pred, _, _ = self.gp.predict(np.array([[z,V]]))
                dxdt[t,n,self.x_offset] = m_pred[0]

        return dxdt

    def resample(self, data=[]):
        """
        Resample the dynamics function given a list of inferred voltage and state trajectories
        """
        # Extract the latent states and voltages
        Xs = []
        Ys = []
        for d in data:
            z = d[:,self.x_offset:self.x_offset][:,None]
            v = d[:,self.parent_compartment.x_offset][:,None]
            Xs.append(np.hstack((z[:-1,:],v[:-1,:])))
            Ys.append(np.hstack((z[1:,:] - z[:-1,:])))

        X = np.vstack(Xs)
        Y = np.vstack(Ys)

        # Set up the sparse GP regression model with the sampled inputs and outputs
        gpr = GPy.models.SparseGPRegression(X, Y, self.kernel, Z=self.Z)

        # HACK: Rather than using a truly nonparametric approach, just sample
        # the GP at the grid of inducing points and interpolate at the GP mean
        self.h = gpr.posterior_samples(self.Z, size=1)

        # HACK: Recreate the GP with the sampled function h
        self.gp = GPy.models.SparseGPRegression(self.Z, self.h, self.kernel, Z=self.Z)

    def plot(self):
        import matplotlib.pyplot as plt
        h2 = self.h.reshape((self.grid,self.grid))
        plt.imshow(h2, extent=(self.V_min, self.V_max, self.z_max, self.z_min))
        plt.ioff()
        plt.show()