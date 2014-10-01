import numpy as np
from scipy.misc import logsumexp

from pybiophys.utils.utils import ibincount, as_matrix, as_sarray

# TODO: Move the proposals, likelihoods, etc to a separate Python package
# since it is shared among multiple projects now
class Proposal(object):
    """
    General wrapper for a proposal distribution. It must support efficient
    log likelihood calculations and sampling.
    Extend this for the proposal of interest
    """
    def sample_next(self, t_prev, Z_prev, t_next):
        """ Sample the next state Z given Z_prev
        """
        pass

    def logp(self, t_prev, Z_prev, t_next, Z_next):
        return -np.Inf

class DynamicalSystemProposal(Proposal):
    def __init__(self, dzdt, noiseclass):
        self.dzdt = dzdt
        self.noisesampler = noiseclass

    def sample_next(self, t_prev, Z_prev, t_next):
        assert t_next >= t_prev
        D,Np = Z_prev.shape
        z = Z_prev + self.dzdt(t_prev, Z_prev) * (t_next-t_prev)

        # Scale down the noise to account for time delay
        dt = self.t[t_next] - self.t[t_prev]
        sig = self.sigma * dt
        noise = self.noisesampler.sample(Np=Np, sigma=sig)
        logp = self.noisesampler.logp(noise)

        state = {'z' : z}
        return z + noise, logp, state

    def logp(self, t_prev, Z_prev, t_next, Z_next, state=None):
        if state is not None and 'z' in state:
            z = state['z']
        else:
            # Recompute
            z = Z_prev + self.dzdt(t_prev, Z_prev) * (t_next - t_prev)
        return self.noisesampler.logp((Z_next-z)/np.sqrt(t_next-t_prev))


class HodgkinHuxleyProposal(Proposal):
    def __init__(self, population, t, inpt, sigma):
        self.population = population
        self.t = t
        self.inpt = inpt

        self.sigma = sigma
        if self.sigma.ndim == 1:
            self.sigma = self.sigma[:,None]

        from distributions import DiagonalGaussianDistribution
        D = self.sigma.size
        self.noiseclass = DiagonalGaussianDistribution(D, 0, sigma)

        # Set the noise to Gaussian
        # TODO: Decide how to set the variance

    def _hh_kinetics(self, index, Z):
        D,Np = Z.shape

        # Get the current input and latent states
        current_inpt = self.inpt[index]
        latent = as_sarray(Z, self.population.latent_dtype)

        # Run the kinetics forward
        dxdt = self.population.kinetics(latent, current_inpt)

        # Wow, this is terrible. Converting to/from numpy struct array and
        # regular arrays, and maintaining the correct shape, is a mega pain in the ass
        return dxdt.view(np.float).reshape((Np,D)).T

    def sample_next(self, curr_index, Z_prev, next_index):
        # assert next_index >= curr_index
        D,Np = Z_prev.shape

        # Propagate forward according to the hodgkin huxley dynamics
        # then add noise, guaranteeing that we stay within the limits
        dt = self.t[next_index] - self.t[curr_index]
        z = Z_prev + self._hh_kinetics(curr_index, Z_prev) * dt

        # Scale down the noise to account for time delay
        # sig = self.sigma * np.sqrt(dt)
        sig = self.sigma * dt

        # TODO: Remove the zeros_like requirement
        noise = self.noiseclass.sample()
        logp = self.noiseclass.logp(noise)

        # Also return extra state to help compute logp
        state = {'z' : z}
        return z + noise*dt, logp, state

    def logp(self, curr_index, Z_prev, next_index, Z_next, state=None):
        # Propagate forward according to the hodgkin huxley dynamics
        # then add noise, guaranteeing that we stay within the limits
        dt = self.t[next_index] - self.t[curr_index]

        # Check if we have the propagated particles
        if state is not None and 'z' in state:
            z = state['z']
        else:
            # Run kinetics and clip to within range
            z = Z_prev + self._hh_kinetics(curr_index, Z_prev) * dt

        noise = (Z_next-z)/dt

        # Scale down the noise to account for time delay
        # import pdb; pdb.set_trace()
        # sig = self.sigma * np.sqrt(dt)
        logp = self.noiseclass.logp(noise)

        # Sum along the D axis
        logp1 = logp.sum(axis=0)

        return logp1


class TruncatedHodgkinHuxleyProposal(Proposal):
    def __init__(self, population, t, inpt, sigma):
        self.population = population
        self.t = t
        self.inpt = inpt

        self.sigma = sigma
        if self.sigma.ndim == 1:
            self.sigma = self.sigma[:,None]

        self.lb = population.latent_lb[:,None]
        self.ub = population.latent_ub[:,None]

        from distributions import TruncatedGaussianDistribution
        self.noiseclass = TruncatedGaussianDistribution()

    def _hh_kinetics(self, index, Z):
        D,Np = Z.shape
        # Get the current input and latent states
        current_inpt = self.inpt[index]
        latent = as_sarray(Z, self.population.latent_dtype)

        # Run the kinetics forward
        dxdt = self.population.kinetics(latent, current_inpt)

        # TODO: Wow, this is terrible. Converting to/from numpy struct array and
        # regular arrays, and maintaining the correct shape, is a mega pain in the ass
        return dxdt.view(np.float).reshape((Np,D)).T

    def sample_next(self, curr_index, Z_prev, next_index):
        # assert next_index >= curr_index
        D,Np = Z_prev.shape

        # Propagate forward according to the hodgkin huxley dynamics
        # then add noise, guaranteeing that we stay within the limits
        dt = self.t[next_index] - self.t[curr_index]
        z = Z_prev + self._hh_kinetics(curr_index, Z_prev) * dt

        z = np.clip(z, self.lb, self.ub)

        noise_lb = self.lb - z
        noise_ub = self.ub - z

        # Scale down the noise to account for time delay
        # sig = self.sigma * np.sqrt(dt)
        sig = self.sigma * dt

        # TODO: Remove the zeros_like requirement
        noise = self.noiseclass.sample(mu=np.zeros_like(z), sigma=sig,
                                       lb=noise_lb, ub=noise_ub)
        logp = self.noiseclass.logp(noise, mu=0, sigma=sig,
                                    lb=noise_lb, ub=noise_ub)

        # Also return extra state to help compute logp
        state = {'z' : z}
        return z + noise, logp, state

    def logp(self, curr_index, Z_prev, next_index, Z_next, state=None):
        # Propagate forward according to the hodgkin huxley dynamics
        # then add noise, guaranteeing that we stay within the limits
        dt = self.t[next_index] - self.t[curr_index]

        # Check if we have the propagated particles
        if state is not None and 'z' in state:
            z = state['z']
        else:
            # Run kinetics and clip to within range
            z = Z_prev + self._hh_kinetics(curr_index, Z_prev) * dt
            z = np.clip(z, self.lb, self.ub)



        noise = Z_next-z
        noise_lb = self.lb - z
        noise_ub = self.ub - z

        # Scale down the noise to account for time delay
        # import pdb; pdb.set_trace()
        # sig = self.sigma * np.sqrt(dt)
        sig = self.sigma * dt
        logp = self.noiseclass.logp(noise, mu=0, sigma=sig,
                                    lb=noise_lb, ub=noise_ub)

        # Sum along the D axis
        logp1 = logp.sum(axis=0)

        # if not np.all(np.isfinite(logp1)):
        #     import pdb; pdb.set_trace()
        return logp1


class Likelihood(object):
    """
    General wrapper for a proposal distribution. It must support efficient
    log likelihood calculations and sampling.
    Extend this for the proposal of interest
    """
    def logp(self, X, Z):
        """ Compute the log probability of X given Z
        """
        return -np.Inf

    def sample(self, Z):
        """
        Sample an observed state given a latent state
        """
        return None

class NoiseLikelihood(Likelihood):
    def __init__(self, noiseclass):
        self.noisesampler = noiseclass

    def logp(self, X, Z):
        # assert X.shape == Z.shape, "ERROR: X and Z must be the same shape"
        return self.noisesampler.logp(X-Z)

    def sample(self, Z):
        """
        Sample an observed state given a latent state
        """
        return Z + self.noisesampler.sample(Np=Z.size)


class ObservationLikelihood(Likelihood):
    """
    Likelihood model where X is a noisy version of only a subset of Z's rows
    """
    def __init__(self,  model):
        self.model = model
        self.lb = model.population.latent_lb[:,None]
        self.ub = model.population.latent_ub[:,None]

    def logp(self, X, Z):
        # View as the model's dtypes
        obs = as_sarray(X, self.model.observation.observed_dtype)
        latent = as_sarray(Z, self.model.population.latent_dtype)
        logp = self.model.observation.logp(latent, obs)

        # Check if any observations are out of the allowable range
        # oob_lb = np.any(Z < self.lb, axis=0)
        oob_lb = np.sum(Z < self.lb, axis=0) > 0
        oob_ub = np.sum(Z > self.ub, axis=0) > 0
        oob = oob_lb | oob_ub
        logp[oob] = -np.inf
        # logp += -np.inf * oob

        return logp

    def sample(self, Z):
        """
        Sample an observed state given a latent state
        """
        latent = Z.ravel('F').view(self.model.population.dtype)
        return self.model.observation.sample(latent)


class PartiallyObservationLikelihood(Likelihood):
    """
    Likelihood model where X is a noisy version of only a subset of Z's rows
    """
    def __init__(self,  inds, noiseclass):
        self.inds = inds
        self.noisesampler = noiseclass

    def logp(self, X, Z):
        # assert X.shape == Z.shape, "ERROR: X and Z must be the same shape"
        return self.noisesampler.logp(X - Z[self.inds,:])

    def sample(self, Z):
        """
        Sample an observed state given a latent state
        """
        return Z[self.inds] + self.noisesampler.sample(Np=Z[self.inds].size)


class VoltageObservationLikelihood(Likelihood):
    """
    Likelihood model where X is a noisy version of the voltage only.
    """
    def __init__(self,  noiseclass):
        self.noisesampler = noiseclass

    def logp(self, X, Z):
        # assert X.shape == Z.shape, "ERROR: X and Z must be the same shape"
        lp = 0
        for compname in Z.dtype.names:
            lp += self.noisesampler.logp(X[compname]['V'] - Z[compname]['V'])
        return lp

    def sample(self, Z):
        """
        Sample an observed state given a latent state
        """
        X = np.zeros_like(Z)
        for compname in Z.dtype.names:
            X[compname]['V'] = Z[compname]['V'] + self.noisesampler.sample(Np=Z.size)

        return X

class ParticleGibbsAncestorSampling(object):
    """
    A conditional particle filter with ancestor sampling
    """
    def __init__(self,
                 D,
                 T,
                 t0,
                 X0,
                 proposal,
                 lkhd,
                 p_initial,
                 fixed_particle,
                 Np=1000):
        """
        Initialize the particle filter with:
        D:           Dimensionality of latent state space
        T:           Number of time steps
        proposal:    A proposal distribution for new particles given old
        lkhd:        An observation likelihood model for particles
        p_initial:   Distribution over initial states
        fixed_particle: Particle representing current state of Markov chain
        Np:          The number of paricles
        """
        # Initialize data structures for particles and weights
        self.D = D
        self.T = T
        self.Np = Np
        self.proposal = proposal
        self.lkhd = lkhd

        # Store the particles in a (D x Np) matrix for fast, vectorized
        # proposals and likelihood evaluations
        self.buffer = T
        self.times = np.zeros(self.buffer)

        # We should probably do this the other way around since
        # T > Np > D in most cases!
        self.particles = np.zeros((D,Np,self.buffer))
        self.ancestors = np.zeros((Np, self.buffer), dtype=np.int)
        self.weights = np.zeros((Np, self.buffer))

        # Keep track of the times when the filter has been called
        self.times[0] = t0

        # Store the fixed particle
        assert fixed_particle.shape == (D,T)
        self.fixed_particle = fixed_particle

        # Let the first particle correspond to the fixed particle
        # Sample the initial state
        self.particles[:,0,0] = self.fixed_particle[:,0]
        self.particles[:,1:,0] = p_initial.sample(Np=self.Np-1)

        # Initialize weights according to observation likelihood
        log_W = self.lkhd.logp(X0, self.particles[:,:,0])
        self.weights[:,0] = np.exp(log_W - logsumexp(log_W))
        self.weights[:,0] /= self.weights[:,0].sum()

        # Increment the offset to point to the next particle slot
        self.offset = 1

    def filter(self,
               t_next,
               X_next,
               resample_method='lowvariance'):
        """
        Filter a given observation sequence to get a sequence of latent states, Z.
        """
        # Save the current filter time
        self.times[self.offset] = t_next

        # First, ressample the previous parents
        curr_ancestors = self._resample(self.weights[:, self.offset-1],
                                        resample_method)
        # TODO Is permutation necessary?
        # curr_ancestors = np.random.permutation(curr_ancestors)

        # Move each particle forward according to the proposal distribution
        prev_particles = self.particles[:,curr_ancestors,self.offset-1]
        new_particles, _, state = self.proposal.sample_next(self.times[self.offset-1],
                                                                     prev_particles,
                                                                     t_next)
        self.particles[:,:, self.offset] = new_particles

        # Override the first particle with the fixed particle
        self.particles[:, 0, self.offset] = self.fixed_particle[:, self.offset]

        # Resample the parent index of the fixed particle
        logw_as = np.log(self.weights[:,self.offset-1]) + \
                  self.proposal.logp(self.times[self.offset-1],
                                     self.particles[:,:,self.offset-1],
                                     t_next,
                                     self.fixed_particle[:,self.offset].reshape(self.D,1),
                                     state=state
                                     )
        w_as = np.exp(logw_as - logsumexp(logw_as))
        w_as /= w_as.sum()

        # import pdb; pdb.set_trace()
        curr_ancestors[0] = np.random.choice(np.arange(self.Np),
                                             p=w_as)
                                             
        # curr_ancestors[0] = np.sum(np.cumsum(w_as) < np.random.rand()) - 1

        # Save the ancestors
        self.ancestors[:, self.offset] = curr_ancestors

        # Update the weights. Since we sample from the prior,
        # the particle weights are always just a function of the likelihood
        log_W = self.lkhd.logp(X_next, self.particles[:,:,self.offset])
        w = np.exp(log_W - logsumexp(log_W))
        w /= w.sum()
        self.weights[:, self.offset] = w

        # Increment the offset
        self.offset += 1

    def get_trajectory(self, i):
        # Compute the i-th trajectory from the particles and ancestors
        T = self.offset

        x = np.zeros((self.D, T))
        x[:,T-1] = self.particles[:,i,T-1]
        curr_ancestor = self.ancestors[i,T-1]

        for t in np.arange(T-1)[::-1]:
            x[:,t] = self.particles[:,curr_ancestor,t]
            curr_ancestor = self.ancestors[curr_ancestor, t]
        return x

    @property
    def trajectories(self):
        # Compute trajectories from the particles and ancestors
        T = self.offset

        if not np.allclose(self.particles[:,0,:T], self.fixed_particle[:,:T]):
            import pdb; pdb.set_trace()

        x = np.zeros((self.D, self.Np, T))
        x[:,:,T-1] = self.particles[:,:,T-1]
        curr_ancestors = self.ancestors[:,T-1]

        for t in np.arange(T-1)[::-1]:
            x[:,:,t] = self.particles[:,curr_ancestors,t]
            curr_ancestors = self.ancestors[curr_ancestors, t]
        return x

    @property
    def trajectory_weights(self):
        return self.weights[:, self.offset-1]

    def sample_trajectory(self):
        # Sample a particular weight trace given the particle weights at time T
        # i = np.sum(np.cumsum(self.trajectory_weights) < np.random.rand()) - 1
        i = np.random.choice(np.arange(self.Np), p=self.trajectory_weights)
        return self.trajectories[:,i,:self.offset].reshape((self.D,self.offset))

    def _resample(self, w, method='lowvariance'):
        # Resample all but the fixed particle
        assert method in ['lowvariance','independent']
        if method is 'lowvariance':
            sources = self._lowvariance_sources(w, self.Np)
        if method is 'independent':
            sources = self._independent_sources(w, self.Np)

        return sources

    def _independent_sources(self, w, num):
        # Return an ordered array of source indices from source counts
        # e.g. if the sources are 3x'0', 2x'1', 0x'2', and 1x'3', as specified
        # by the vector [3,2,0,1], then the output will be
        # [0, 0, 0, 1, 1, 3]
        return ibincount(np.random.multinomial(num,w))

    def _lowvariance_sources(self, w, num):
        r = np.random.rand()/num
        bins = np.concatenate(((0,),np.cumsum(w)))
        return ibincount(np.histogram(r+np.linspace(0,1,num,endpoint=False), bins)[0])

