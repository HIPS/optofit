
import itertools
import numpy as np

from channels import Channel

from GPy.core.sparse_gp import SparseGP
from GPy.models import SparseGPRegression, GPRegression
from GPy.kern import rbf
from GPy.likelihoods import Gaussian

import matplotlib.pyplot as plt

sigma = lambda x: 1./(1+np.exp(-x))
sigma_inv = lambda x: np.log(x/(1-x))

class SparseGPWithVariance(SparseGP):
    def __init__(self, X, Y, kernel, Z, Y_variance):
        # likelihood defaults to Gaussian
        likelihood = Gaussian(Y, variance=Y_variance)

        SparseGP.__init__(self, X, likelihood, kernel, Z=Z)
        self.ensure_default_constraints()


class GPChannel(Channel):
    """
    Generic channel with a Gaussian process dyamics function
    """
    def __init__(self, name=None, hypers=None):
        super(GPChannel, self).__init__(name=name)

        # Specify the number of parameters
        self.D = hypers['D']
        # Channels are expected to have a n_x variable
        self.n_x = self.D

        # Set the hyperparameters
        self.g = hypers['g_gp']
        self.E = hypers['E_gp']
        self.a0 = hypers['alpha_0']
        self.b0 = hypers['beta_0']
        self.sigmas = (self.a0/self.b0) * np.ones(self.D)

        # Create a GP Object for the dynamics model
        # This is a function Z x V -> dZ, i.e. a 2D
        # Gaussian process regression.
        # Lay out a grid of inducing points for a sparse GP
        self.grid = 5
        self.z_min = sigma_inv(0.001)
        self.z_max = sigma_inv(0.999)
        self.V_min = -80.
        self.V_max = 50.
        length = 5.0
        self.Z = np.array(list(
                      itertools.product(*([np.linspace(self.z_min, self.z_max, self.grid) for _ in range(self.D)]
                                        + [np.linspace(self.V_min, self.V_max, self.grid)]))))

        # Create independent RBF kernels over Z and V
        kernel_z_hyps = { 'input_dim' : 1,
                          'variance' : hypers['sigma_kernel'],
                          'lengthscale' : (self.z_max-self.z_min)/length}

        self.kernel_z = rbf(**kernel_z_hyps)
        for n in range(1, self.D):
            self.kernel_z = self.kernel_z.prod(rbf(**kernel_z_hyps), tensor=True)

        kernel_V_hyps = { 'input_dim' : 1,
                          'variance' : hypers['sigma_kernel'],
                          'lengthscale' : (self.V_max-self.V_min)/length}
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

        self.hs = []
        self.gps = []

        for d in range(self.D):
            # Sample the function value at the inducing points
            self.hs.append(np.random.multivariate_normal(m, C, 1).T)

            # Create a sparse GP model with the sampled function h
            # This will be used for prediction
            self.gps.append(SparseGPWithVariance(self.Z, self.hs[d], self.kernel, self.Z, self.sigmas[d]))


    def steady_state(self, x0):
        V = x0[self.parent_compartment.x_offset]

        # TODO: Set the steady state
        x0[self.x_offset] = sigma_inv(0.005)

    def current(self, x, V, t, n):
        """
        Evaluate the instantaneous current through this channel
        """
        z = x[t,n,self.x_offset]
        return sigma(z) * (V - self.E)

    def kinetics(self, dxdt, x, inpt, ts):
        #import pdb; pdb.set_trace()
        N = x.shape[1]
        S = ts.shape[0]

        # Get a pointer to the voltage of the parent compartment
        # TODO: This approach sucks b/c it assumes the voltage
        # is the first compartment state. It should be faster than
        # calling back into the parent to have it extract the voltage
        # for us though.
        for s in range(S):
            t = ts[s]
            V = x[t,:,self.parent_compartment.x_offset]
            z = x[t,:,self.x_offset:self.x_offset+self.D]
            zz = np.hstack((np.reshape(z,(N,self.D)), np.reshape(V, (N,1))))

            # Sample from the GP kinetics model
            for d in range(self.D):
                m_pred, v_pred, _, _ = self.gps[d].predict(zz)

                if self.model_dzdt:
                    dxdt[t,:,self.x_offset+d] = m_pred[:,0]
                else:
                    dxdt[t,:,self.x_offset+d] = m_pred[:,0] - z[:,d]

        return dxdt

    def _compute_regression_data(self, datas, dts, d=0):
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
            z = data[:,self.x_offset:self.x_offset+self.D]
            v = data[:,self.parent_compartment.x_offset][:,None]

            Xs.append(np.hstack((z[:-1,:],v[:-1,:])))

            if self.model_dzdt:
                ddt = dt.reshape((z.shape[0]-1, 1))
                dz = (z[1:,d:d+1] - z[:-1,d:d+1]) / ddt
                Ys.append(dz)
            else:
                Ys.append(z[1:,d:d+1])

        X = np.vstack(Xs)
        Y = np.vstack(Ys)

        return X,Y

    def resample(self, data=[], dt=1):
        """
        Resample the dynamics function given a list of inferred voltage and state trajectories
        """
        for d in range(self.D):
            # Compute the regression data with dt
            X,Y = self._compute_regression_data(data, dt, d=d)

            # Set up the sparse GP regression model with the sampled inputs and outputs
            gpr = SparseGPWithVariance(X, Y, self.kernel, self.Z, self.sigmas[d])
            # gpr = GPRegression(X, Y, self.kernel)

            # HACK: Rather than using a truly nonparametric approach, just sample
            # the GP at the grid of inducing points and interpolate at the GP mean
            h = gpr.posterior_samples_f(self.Z, size=1)

            # HACK: Rather than sampling, just use the predicted mean. There seems to be
            # way too much variance in the samples
            # h,_,_,_ = gpr.predict(self.Z)

            # HACK: Recreate the GP with the sampled function h
            gp = SparseGPWithVariance(self.Z, h, self.kernel, self.Z, self.sigmas[d])

            self.hs[d] = h
            self.gps[d] = gp

    def resample_transition_noise(self, data, t):
        """
        Resample sigma, the transition noise variance, under an inverse gamma prior
        """
        #raise NotImplementedError("Still need to implement gamma resampling")

        # Compute the predicted state and the actual next state
        Xs = []
        X_preds = []
        X_diffs = []
        
        T = data.shape[0]
        D = data.shape[1]
        dxdt = np.zeros((T,1,D))
        x = np.zeros((T,1,D))
        x[:,0,:] = data

        # Compute kinetics with no input
        inpt = None
        dxdt = self.kinetics(dxdt, x, inpt, np.arange(T-1))
        dt = np.diff(t)
        
        #import pdb; pdb.set_trace()
        # Compute predicted state given kinetics
        X_pred = data[:-1,self.x_offset:self.x_offset+self.D,] + \
                 dxdt[:,0,:][:-1,self.x_offset:self.x_offset+self.D]*np.dot(np.ones((self.D, 1)), [dt]).T
        X = data[1:, self.x_offset:self.x_offset+self.D]

        X_diff = X_pred - X
        
        Xs.append(X)
        X_preds.append(X_pred)
        X_diffs.append(X_diff)

        # TODO: Resample transition noise. See Wikipedia for form of posterior of normal-gamma model
        X_diffs = np.array(X_diffs)
        for d in range(self.D):
            #import pdb; pdb.set_trace()
            alpha = self.a0 + len(X_diffs[0,:,d]) / 2.0
            beta  = self.b0 + np.sum(X_diffs[0,:,d] ** 2) / 2.0
            self.sigmas[d] = beta / alpha

    def plot(self, ax=None, im=None, l=None, cmap=plt.cm.hot, data=[]):
        if self.D > 1:
            print "Can only plot 1D GP models"
            return

        # Reshape into a 2D function image
        h2 = self.hs[0].reshape((self.grid,self.grid))

        if im is None and ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            im = ax.imshow(h2, extent=(self.V_min, self.V_max, self.z_max, self.z_min), cmap=cmap, vmin=-3, vmax=3)
            ax.set_aspect((self.V_max-self.V_min)/(self.z_max-self.z_min))
            ax.set_ylabel('z')
            ax.set_xlabel('V')
            ax.set_title('dz_%s/dt(z,V)' % self.name)

        elif im is None and ax is not None:
            im = ax.imshow(h2, extent=(self.V_min, self.V_max, self.z_max, self.z_min), cmap=cmap, vmin=-3, vmax=3)
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

class GPKdrChannel(GPChannel):
    """
    Resample h with the true transition model
    """
    # def kinetics(self, dxdt, x, inpt, ts):
    #     # TODO: DEBUG!!!!!!!!!!!!!!!!!!!!!
    #     T = x.shape[0]
    #     N = x.shape[1]
    #     D = x.shape[2]
    #     M = inpt.shape[1]
    #     S = ts.shape[0]
    #
    #     # Evaluate dz/dt using true Kdr dynamics
    #     g = lambda x: x**4
    #     ginv = lambda u: u**(1./4)
    #     dg_dx = lambda x: 4*x**3
    #
    #     logistic = sigma
    #     dlogit = lambda x: 1./(x*(1.0-x))
    #     u_to_x = lambda u: ginv(logistic(u))
    #
    #     # Compute dynamics du/dt
    #     alpha = lambda V: 0.01*(V+55.)/(1-np.exp(-(V+55.)/10.))
    #     beta = lambda V: 0.125*np.exp(-(V+65.)/80.)
    #     dx_dt = lambda x,V: alpha(V)*(1-x) - beta(V) * x
    #     du_dt = lambda u,V: dlogit(g(u_to_x(u))) * dg_dx(u_to_x(u)) * dx_dt(u_to_x(u),V)
    #
    #     # Get a pointer to the voltage of the parent compartment
    #     # TODO: This approach sucks b/c it assumes the voltage
    #     # is the first compartment state. It should be faster than
    #     # calling back into the parent to have it extract the voltage
    #     # for us though.
    #     for s in range(S):
    #         t = ts[s]
    #         V = x[t,:,self.parent_compartment.x_offset]
    #         z = x[t,:,self.x_offset]
    #         dxdt[t,:,self.x_offset] = du_dt(np.asarray(z),np.asarray(V))
    #
    #     return dxdt

    def given_resample(self, data=[], dt=1):
        """
        Resample the dynamics function given a list of inferred voltage and state trajectories
        """
        # import pdb; pdb.set_trace()
        uu = self.Z[:,0]
        V = self.Z[:,1]

        # Evaluate dz/dt using true Kdr dynamics
        g = lambda x: x**4
        ginv = lambda u: u**(1./4)
        dg_dx = lambda x: 4*x**3

        logistic = sigma
        logit = sigma_inv
        dlogit = lambda x: 1./(x*(1.0-x))

        u_to_x = lambda u: ginv(logistic(u))
        x_to_u = lambda x: logit(g(x))
        # uu = x_to_u(xx)

        # Compute dynamics du/dt
        alpha = lambda V: 0.01*(V+55.)/(1-np.exp(-(V+55.)/10.))
        beta = lambda V: 0.125*np.exp(-(V+65.)/80.)
        dx_dt = lambda x,V: alpha(V)*(1-x) - beta(V) * x
        du_dt = lambda u,V: dlogit(g(u_to_x(u))) * dg_dx(u_to_x(u)) * dx_dt(u_to_x(u),V)

        X = self.Z
        Y = du_dt(uu, V)[:,None]

        # Set up the sparse GP regression model with the sampled inputs and outputs
        # gpr = SparseGPRegression(X, Y, self.kernel, Z=self.Z)
        # gpr.likelihood.variance = 0.01
        gpr = GPRegression(X, Y, self.kernel)

        # HACK: Optimize the hyperparameters
        # contrain all parameters to be positive
        # gpr.constrain_positive('')

        # optimize and plot
        # gpr.ensure_default_constraints()
        # gpr.optimize_restarts(num_restarts=10)
        # gpr.plot()
        # import pdb; pdb.set_trace()

        # HACK: Rather than using a truly nonparametric approach, just sample
        # the GP at the grid of inducing points and interpolate at the GP mean
        self.h = gpr.posterior_samples_f(self.Z, size=1)

        # HACK: Recreate the GP with the sampled function h
        self.gp = SparseGPRegression(self.Z, self.h, self.kernel, Z=self.Z)
