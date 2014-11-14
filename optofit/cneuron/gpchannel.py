
import itertools
import numpy as np

from channels import Channel

from GPy.models import SparseGPRegression, GPRegression
from GPy.kern import rbf


import matplotlib.pyplot as plt

sigma = lambda x: 1./(1+np.exp(-x))
sigma_inv = lambda x: np.log(x/(1-x))

class GPChannel(Channel):
    """
    Generic channel with a Gaussian process dyamics function
    """
    def __init__(self, name=None, hypers=None):
        super(GPChannel, self).__init__(name=name)

        # Specify the number of parameters
        self.n_x = 1

        # Set the hyperparameters
        self.g = hypers['g_gp']
        self.E = hypers['E_gp']

        # Create a GP Object for the dynamics model
        # This is a function Z x V -> dZ, i.e. a 2D
        # Gaussian process regression.
        # Lay out a grid of inducing points for a sparse GP
        self.grid = 10
        self.z_min = sigma_inv(0.001)
        self.z_max = sigma_inv(0.999)
        self.V_min = -80.
        self.V_max = 120.
        self.Z = np.array(list(
                      itertools.product(*([np.linspace(self.z_min, self.z_max, self.grid) for _ in range(self.n_x)]
                                        + [np.linspace(self.V_min, self.V_max, self.grid)]))))

        # Create independent RBF kernels over Z and V
        kernel_z_hyps = { 'input_dim' : 1,
                          'variance' : hypers['sig'],
                          'lengthscale' : (self.z_max-self.z_min)/4.}

        self.kernel_z = rbf(**kernel_z_hyps)
        for n in range(1, self.n_x):
            self.kernel_z = self.kernel_z.prod(rbf(kernel_z_hyps))

        kernel_V_hyps = { 'input_dim' : 1,
                          'variance' : hypers['sig'],
                          'lengthscale' : (self.V_max-self.V_min)/4.}
        self.kernel_V = rbf(**kernel_V_hyps)

        # Combine the kernel for z and V
        self.kernel = self.kernel_z.prod(self.kernel_V, tensor=True)

        # Initialize with a random sample from the prior
        self.model_dzdt = True
        if self.model_dzdt:
            m = np.zeros(self.Z.shape[0])
        else:
            m = self.Z[:,0]
        C = self.kernel.K(self.Z)
        self.h = np.random.multivariate_normal(m, C, 1).T

        # Create a sparse GP model with the sampled function h
        # This will be used for prediction
        self.gp = SparseGPRegression(self.Z, self.h, self.kernel, Z=self.Z)


    def steady_state(self, x0):
        V = x0[self.parent_compartment.x_offset]

        # TODO: Set the steady state
        x0[self.x_offset] = sigma_inv(0.005)

    #cpdef double current(self, double[:,:,::1] x, double V, int t, int n):
    def current(self, x, V, t, n):
        """
        Evaluate the instantaneous current through this channel
        """
        #cdef double z = x[t,n,self.x_offset]
        z = x[t,n,self.x_offset]
        return sigma(z) * (V - self.E)

    def kinetics(self, dxdt, x, inpt, ts):
        T = x.shape[0]
        N = x.shape[1]
        D = x.shape[2]
        M = inpt.shape[1]
        S = ts.shape[0]

        # Get a pointer to the voltage of the parent compartment
        # TODO: This approach sucks b/c it assumes the voltage
        # is the first compartment state. It should be faster than
        # calling back into the parent to have it extract the voltage
        # for us though.
        for s in range(S):
            t = ts[s]
            V = x[t,:,self.parent_compartment.x_offset]
            z = x[t,:,self.x_offset]
            zz = np.hstack((np.reshape(z,(N,1)), np.reshape(V, (N,1))))

            # Sample from the GP kinetics model
            m_pred, v_pred, _, _ = self.gp.predict(zz)

            if self.model_dzdt:
                dxdt[t,:,self.x_offset] = m_pred[:,0]
            else:
                dxdt[t,:,self.x_offset] = m_pred[:,0] - z

        return dxdt

    def _compute_regression_data(self, datas, dts):
        # Make sure d is a list
        assert isinstance(datas, list) or isinstance(datas, np.ndarray)
        if isinstance(datas, np.ndarray):
            datas = [datas]

        assert isinstance(dts, list) or isinstance(dts, np.ndarray) or np.isscalar(dts)
        if isinstance(dts, np.ndarray):
            dts = [dts]
        elif np.isscalar(dts):
            dts = [dts * np.ones((data.shape[0]-1,1)) for data in datas]

        # Extract the latent states and voltages
        Xs = []
        Ys = []
        for data, dt in zip(datas, dts):
            z = data[:,self.x_offset][:,None]
            v = data[:,self.parent_compartment.x_offset][:,None]

            Xs.append(np.hstack((z[:-1,:],v[:-1,:])))

            if self.model_dzdt:
                ddt = dt.reshape((z.shape[0]-1, 1))
                dz = (z[1:,:] - z[:-1,:]) / ddt
                Ys.append(dz)
            else:
                Ys.append(z[1:,:])

        X = np.vstack(Xs)
        Y = np.vstack(Ys)

        return X,Y

    def resample(self, data=[], dt=1):
        """
        Resample the dynamics function given a list of inferred voltage and state trajectories
        """
        # TODO: Get dt
        X,Y = self._compute_regression_data(data, dt)

        # Set up the sparse GP regression model with the sampled inputs and outputs
        gpr = SparseGPRegression(X, Y, self.kernel, Z=self.Z)
        # gpr = GPRegression(X, Y, self.kernel)


        # HACK: Rather than using a truly nonparametric approach, just sample
        # the GP at the grid of inducing points and interpolate at the GP mean
        self.h = gpr.posterior_samples_f(self.Z, size=1)

        # HACK: Recreate the GP with the sampled function h
        self.gp = SparseGPRegression(self.Z, self.h, self.kernel, Z=self.Z)

    def plot(self, ax=None, im=None, l=None, cmap=plt.cm.hot, data=[]):

        # Reshape into a 2D function image
        h2 = self.h.reshape((self.grid,self.grid))

        if im is None and ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            im = ax.imshow(h2, extent=(self.V_min, self.V_max, self.z_max, self.z_min), cmap=cmap, vmin=-15, vmax=15)
            ax.set_aspect((self.V_max-self.V_min)/(self.z_max-self.z_min))
            ax.set_ylabel('z')
            ax.set_xlabel('V')
            ax.set_title('dz_%s/dt(z,V)' % self.name)

        elif im is None and ax is not None:
            im = ax.imshow(h2, extent=(self.V_min, self.V_max, self.z_max, self.z_min), cmap=cmap, vmin=-15, vmax=15)
            ax.set_aspect((self.V_max-self.V_min)/(self.z_max-self.z_min))
            ax.set_ylabel('z')
            ax.set_xlabel('V')
            ax.set_title('dz_%s/dt(z,V)' % self.name)

        elif im is not None:
            im.set_data(h2)

        if len(data) > 0:
            X,Y = self._compute_regression_data(data, dts=1)
            if l is None and ax is not None:
                l = ax.plot(X[:,1], X[:,0], 'ko')
            elif l is not None:
                l[0].set_data(X[:,1], X[:,0])

        if ax is not None:
            ax.set_xlim([self.V_min, self.V_max])
            ax.set_ylim([self.z_max, self.z_min])


        return ax, im, l