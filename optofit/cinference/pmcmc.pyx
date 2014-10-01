# distutils: extra_compile_args = -O3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
## cython: cdivision=True
from cython.parallel import prange

import numpy as np
cimport numpy as np

from hips.inference.particle_mcmc cimport InitialDistribution, Proposal, Likelihood, ParticleGibbsAncestorSampling
from optofit.cneuron.component cimport Component

class GaussianInitialDistribution(InitialDistribution):

    def __init__(self, mu, sigma):
        # Check sizes
        if np.isscalar(mu) and np.isscalar(sigma):
            self.D = 1
            mu = np.atleast_2d(mu)
            sigma = np.atleast_2d(sigma)

        elif mu.ndim == 1 and sigma.ndim == 2:
            assert mu.shape[0] == sigma.shape[0] == sigma.shape[1]
            self.D = mu.shape[0]
            mu = mu.reshape((1,self.D))

        elif mu.ndim == 2 and sigma.ndim == 2:
            assert mu.shape[1] == sigma.shape[0] == sigma.shape[1] and mu.shape[0] == 1
            self.D = mu.shape[1]
        else:
            raise Exception('Invalid shape for mu and sigma')

        self.mu = mu
        self.sigma = sigma
        self.chol = np.linalg.cholesky(self.sigma)

    def sample(self, N=1):
        smpls = np.tile(self.mu, (N,1))
        smpls += np.dot(np.random.randn(N,self.D), self.chol)
        return smpls


cdef class HodgkinHuxleyProposal(Proposal):
    # Latent state space dimensionality
    cdef int D
    # Transition model
    cdef Component component
    # Transition noise for each dimension
    cdef double[::1] sigmas
    cdef double[::1] sigma_sqs
    # Times at which the proposal will be requested
    cdef double[::1] ts
    # The input as a function of time
    cdef double[:,::1] inpt
    # A buffer for kinetics
    cdef double[:,:,::1] dzdt

    def __init__(self, int T, int N, int D, Component component, double[::1] sigmas, double[::1] ts, double[:,::1] inpt):
        self.component = component
        self.sigmas = sigmas
        self.ts = ts
        self.inpt = inpt

        # Precompute sigma**2
        cdef int d
        self.D = D
        self.sigma_sqs = np.zeros(self.D)
        for d in range(self.D):
            self.sigma_sqs[d] = self.sigmas[d]**2

        # Allocate space for dzdt
        self.dzdt = np.empty((T,N,D))

    cpdef sample_next(self, double[:,:,::1] z, int i_prev, int[::1] ancestors):
        """ Sample the next state given the previous time index

            :param z:       TxNxD buffer of particle states
            :param i_prev:  Time index into z and self.ts

            :return         z[i_prev+1,:,:] is updated with a sample
                            from the proposal distribution.
        """
        cdef int N = z.shape[1]
        cdef int D = z.shape[2]
        cdef int n, d

        # Preallocate random variables
        cdef double[:,::1] rands = np.random.randn(N,D)

        # Run the kinetics model forward
        cdef int[::1] tview = <int[:1]> &i_prev
        self.component.kinetics(self.dzdt, z, self.inpt, tview)
        cdef double dt = self.ts[i_prev+1]-self.ts[i_prev]

        # # TODO: Parallelize with OMP
        with nogil:
            for n in prange(N):
                for d in prange(D):
                    # Forward Euler step
                    z[i_prev+1,n,d] = z[i_prev,n,d] + dt * self.dzdt[i_prev,n,d]
                    # Add noise
                    z[i_prev+1,n,d] += self.sigmas[d] * rands[n,d]


    cpdef logp(self, double[:,::1] z_prev, int i_prev, double[::1] z_curr, double[::1] lp):
        """ Compute the log probability of transitioning from z_prev to z_curr
            at time self.ts[i_prev] to self.ts[i_prev+1]

            :param z_prev:  NxD buffer of particle states at the i_prev-th time index
            :param i_prev:  Time index into self.ts
            :param z_curr:  D buffer of particle states at the (i_prev+1)-th time index
            :param lp:      NxM buffer in which to store the probability of each transition

            :return         z[i_prev+1,:,:] is updated with a sample
                            from the proposal distribution.
        """
        cdef int N = z_prev.shape[0]
        cdef int D = z_prev.shape[1]
        cdef int n, d
        cdef double[::1] z_mean = np.zeros((D,))
        cdef double dt = self.ts[i_prev+1]-self.ts[i_prev]

        # NOTE! We are assuming that dzdt has already been properly populated!
        #
        # # TODO: Parallelize with OMP
        with nogil:
            for n in prange(N):
                for d in prange(D):
                    # Forward Euler step
                    z_mean[d] = z_prev[n,d] + dt * self.dzdt[i_prev,n,d]

                # Compute the Gaussian log probability
                lp[n] = 0
                for d in range(D):
                    lp[n] += -0.5/self.sigma_sqs[d] * (z_curr[d] - z_mean[d])**2


# class TruncatedHodgkinHuxleyProposal(Proposal):
#     def __init__(self, population, t, inpt, sigma):
#         self.population = population
#         self.t = t
#         self.inpt = inpt
#
#         self.sigma = sigma
#         if self.sigma.ndim == 1:
#             self.sigma = self.sigma[:,None]
#
#         self.lb = population.latent_lb[:,None]
#         self.ub = population.latent_ub[:,None]
#
#         from distributions import TruncatedGaussianDistribution
#         self.noiseclass = TruncatedGaussianDistribution()
#
#     def _hh_kinetics(self, index, Z):
#         D,Np = Z.shape
#         # Get the current input and latent states
#         current_inpt = self.inpt[index]
#         latent = as_sarray(Z, self.population.latent_dtype)
#
#         # Run the kinetics forward
#         dxdt = self.population.kinetics(latent, current_inpt)
#
#         # TODO: Wow, this is terrible. Converting to/from numpy struct array and
#         # regular arrays, and maintaining the correct shape, is a mega pain in the ass
#         return dxdt.view(np.float).reshape((Np,D)).T
#
#     def sample_next(self, curr_index, Z_prev, next_index):
#         # assert next_index >= curr_index
#         D,Np = Z_prev.shape
#
#         # Propagate forward according to the hodgkin huxley dynamics
#         # then add noise, guaranteeing that we stay within the limits
#         dt = self.t[next_index] - self.t[curr_index]
#         z = Z_prev + self._hh_kinetics(curr_index, Z_prev) * dt
#
#         z = np.clip(z, self.lb, self.ub)
#
#         noise_lb = self.lb - z
#         noise_ub = self.ub - z
#
#         # Scale down the noise to account for time delay
#         # sig = self.sigma * np.sqrt(dt)
#         sig = self.sigma * dt
#
#         # TODO: Remove the zeros_like requirement
#         noise = self.noiseclass.sample(mu=np.zeros_like(z), sigma=sig,
#                                        lb=noise_lb, ub=noise_ub)
#         logp = self.noiseclass.logp(noise, mu=0, sigma=sig,
#                                     lb=noise_lb, ub=noise_ub)
#
#         # Also return extra state to help compute logp
#         state = {'z' : z}
#         return z + noise, logp, state
#
#     def logp(self, curr_index, Z_prev, next_index, Z_next, state=None):
#         # Propagate forward according to the hodgkin huxley dynamics
#         # then add noise, guaranteeing that we stay within the limits
#         dt = self.t[next_index] - self.t[curr_index]
#
#         # Check if we have the propagated particles
#         if state is not None and 'z' in state:
#             z = state['z']
#         else:
#             # Run kinetics and clip to within range
#             z = Z_prev + self._hh_kinetics(curr_index, Z_prev) * dt
#             z = np.clip(z, self.lb, self.ub)
#
#         noise = Z_next-z
#         noise_lb = self.lb - z
#         noise_ub = self.ub - z
#
#         # Scale down the noise to account for time delay
#         # import pdb; pdb.set_trace()
#         # sig = self.sigma * np.sqrt(dt)
#         sig = self.sigma * dt
#         logp = self.noiseclass.logp(noise, mu=0, sigma=sig,
#                                     lb=noise_lb, ub=noise_ub)
#
#         # Sum along the D axis
#         logp1 = logp.sum(axis=0)
#
#         # if not np.all(np.isfinite(logp1)):
#         #     import pdb; pdb.set_trace()
#         return logp1

cdef class PartialGaussianLikelihood(Likelihood):
    """
    Likelihood in which we only observe some subset of the inputs
    """
    cdef int[::1] observed_dims
    cdef int O
    cdef double[::1] etas
    cdef double[::1] eta_sqs

    # A simple (albeit hacky) observation model.
    # We see some set of indices
    def __init__(self, int[::1] observed_dims, double[::1] etas):

        self.observed_dims = observed_dims
        self.O = observed_dims.shape[0]
        self.etas = etas

        cdef int o
        self.eta_sqs = np.zeros(self.O)
        for o in range(self.O):
            self.eta_sqs[o] = etas[o]**2

    cpdef logp(self, double[:,:,::1] z, double[:,::1] x, int i, double[::1] ll):
        """ Compute the log likelihood, log p(x|z), at time index i and put the
            output in the buffer ll.

            :param z:   TxNxD buffer of latent states
            :param x:   TxO buffer of observations
            :param i:   Time index at which to compute the log likelihood
            :param ll:  N buffer to populate with log likelihoods

            :return     Buffer ll should be populated with the log likelihood of
                        each particle.
        """
        cdef int N = z.shape[1]
        cdef int n, o, d
        for n in range(N):
            ll[n] = 0
            for o in range(self.O):
                d = self.observed_dims[o]
                ll[n] += -0.5/self.eta_sqs[o] * (x[i,o] - z[i,n,d])**2

    cpdef sample(self, double[:,:,::1] z, double[:,::1] x, int i, int n):
        """ Sample the next state given the previous time index

            :param z:       TxNxD buffer of particle states
            :param i_prev:  Time index into z and self.ts

            :return         z[i_prev+1,:,:] is updated with a sample
                            from the proposal distribution.
        """
        cdef int o, d
        for o in range(self.O):
            d = self.observed_dims[o]
            x[i,o] = z[i,n,d] + self.etas[o] * np.random.randn()

