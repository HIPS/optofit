"""
Each class implements a MCMC transition
"""
import numpy as np
from hips.inference.ars import adaptive_rejection_sample
from hips.inference.hmc import hmc

from pybiophys.inference.particle_mcmc import *
from pybiophys.inference.distributions import *
from pybiophys.utils.utils import get_item_at_path, as_matrix, as_sarray, sz_dtype
from pybiophys.models.hyperparameters import hypers

import pdb

class MetropolisHastingsUpdate(object):
    """
    Base class for MH updates. Each update targets a specific model component
    and requires certain configuration. For example, an update for the standard GLM
    might require differentiable parameters. Typical updates include:
        - Gibbs updates (sample from conditional distribution)
        - Hamiltonian Monte Carlo (uses gradient info to sample unconstrained cont. vars)
        - Slice sampling (good for correlaed multivariate Gaussians)
    """
    def __init__(self):
        self._target_components = []

    @property
    def target_components(self):
        # Return a list of components that this update applies to
        return self._target_components

    @property
    def target_variables(self):
        # Return a list of variables that this update applies to
        return []


    def preprocess(self):
        """ Do any required preprocessing
        """
        pass

    def update(self, model, cache=None):
        """ Take a MH step starting from the current state
        """

class ConductanceUpdate(MetropolisHastingsUpdate):
    """
    Update the conductances of a neuron
    """
    def __init__(self, compartment):
        self.neuron_name = compartment.parent.name
        self.compartment_name = compartment.name

    def get_compartment(self, model):
        for neuron in model.population.neurons:
            if neuron.name == self.neuron_name:
                break

        for compartment in neuron.compartments:
            if compartment.name == self.compartment_name:
                return compartment

        raise Exception('Could not find compartment: %s' % self.compartment_name)

    def update(self, model, cache=None):
        """
        Sample the conductances of the neuron given its latent state variables
        """
        # c = self.compartment
        c = self.get_compartment(model)
        # Get the sub-structured arrays for this comp
        chs = c.channels

        # Get a list of conductances for this compartment
        gs = [ch.g for ch in chs]
        gsc = np.array([g.value for g in gs])

        dVc_dts = []
        Iscs = []
        for data in model.data_sequences:
            t = data.t
            T = data.T
            inpt = data.input
            state = data.states

            s_comp = get_item_at_path(state, c.path)
            i_comp = get_item_at_path(inpt, c.path)
            Vc = s_comp['V']
            dVc_dt = c.C.value * self.gradient_V(t, Vc)
            dt = np.concatenate((np.diff(t), np.array([t[-1]-t[-2]])))

            # Some of this change is due to injected current
            dVc_dt -= i_comp['I'][:-1]

            dVc_dts.append(dVc_dt)

            # Now compute the per channel currents in this compartment
            Isc = np.zeros((T-1,0))
            for ch in chs:
                if s_comp[ch.name].dtype.names is not None and \
                   'I' in s_comp[ch.name].dtype.names:
                    Isc = np.concatenate((Isc, -1.0*s_comp[ch.name]['I'][:-1,np.newaxis]),
                                         axis=1)

            Iscs.append(Isc)

        # Concatenate the dVs and the currents together
        dVc_dt = np.concatenate(dVc_dts, axis=0)
        Isc = np.concatenate(Iscs, axis=0)

        # Sample each conductance in turn given that
        # C*dVc_dt ~ N(I_in - np.dot(gsc, Isc), sig_V^2)
        for (i,ch) in enumerate(chs):
            i_rem = np.concatenate((np.arange(i), np.arange(i+1,len(chs))))
            gsc_rem = gsc[i_rem]
            Isc_rem = Isc[:,i_rem]

            # prior = gamma(alphas[ch], scale=1.0/betas[ch])
            prior = ch.g.distribution

            dV_resid = dVc_dt - np.dot(Isc_rem, gsc_rem).ravel()

            from scipy.stats import norm
            sig_V = np.asscalar(hypers['sig_V'].value)
            # lkhd = norm(dV_resid[:,np.newaxis], sig_V)

            lkhd = lambda g: (-0.5/sig_V**2 * (np.dot(Isc[:,i][:,np.newaxis], g) - dV_resid[:,np.newaxis])**2).sum(axis=0)

            def _logp(g):
                sh = g.shape
                g2 = np.atleast_2d(g)
                lp = prior.logp(g2)
                # lp += lkhd.logpdf(np.dot(Isc[:,i][:, np.newaxis], g2)).sum(0)
                lp += lkhd(g2)
                return lp.reshape(sh)

            # Evaluate at a few points around the prior mode to seed the ARS
            xs = np.array([.0001, .001, .01, .05, .1, 1.0, 2.0, 4.0, 7.0, 10.0])
            v_xs = _logp(xs)
            assert np.all(np.isfinite(v_xs))

            # Try for a few attempts to find the right bound
            gsc[i] = adaptive_rejection_sample(_logp, xs, v_xs, [ch.g.lower_bound, ch.g.upper_bound])
            
            # Update the channel conductance parameter
            ch.g.value = gsc[i]

    def gradient_V(self, t, V):
        """
        Compute the gradient of V
        """
        # Compute dV/dt using first order differences
        # T = V.size
        # dVdt = np.zeros((T-1,))
        dVdt = (V[1:]-V[:-1])/(t[1:]-t[:-1])
        # dVdt[-1] = (V[-1]-V[-2])/(t[-1]-t[-2])
        return dVdt


class HmcConductanceUpdate(MetropolisHastingsUpdate):
    """
    Update the conductances of a neuron
    """
    def __init__(self, compartment):
        self.neuron_name = compartment.parent.name
        self.compartment_name = compartment.name

        C = len(compartment.channels)
        self.avg_accept_rate = 0.9*np.ones(C)
        self.step_sz = 0.01*np.ones(C)

    def get_compartment(self, model):
        for neuron in model.population.neurons:
            if neuron.name == self.neuron_name:
                break

        for compartment in neuron.compartments:
            if compartment.name == self.compartment_name:
                return compartment

        raise Exception('Could not find compartment: %s' % self.compartment_name)

    def get_dV_and_currents(self, model):
        c = self.get_compartment(model)
        # Get the sub-structured arrays for this comp
        chs = c.channels

        dts = []
        dVc_dts = []
        Iscs = []
        for data in model.data_sequences:
            t = data.t
            T = data.T
            inpt = data.input
            state = data.states

            s_comp = get_item_at_path(state, c.path)
            i_comp = get_item_at_path(inpt, c.path)
            Vc = s_comp['V']
            dVc_dt = c.C.value * self.gradient_V(t, Vc)
            dt = np.concatenate((np.diff(t), np.array([t[-1]-t[-2]])))

            # Some of this change is due to injected current
            dVc_dt -= i_comp['I']

            dts.append(dt)
            dVc_dts.append(dVc_dt)

            # Now compute the per channel currents in this compartment
            Isc = np.zeros((T,0))
            for ch in chs:
                if s_comp[ch.name].dtype.names is not None and \
                   'I' in s_comp[ch.name].dtype.names:
                    Isc = np.concatenate((Isc, -1.0*s_comp[ch.name]['I'][:,np.newaxis]),
                                         axis=1)

            Iscs.append(Isc)

        # Concatenate the dVs and the currents together
        dt = np.concatenate(dts, axis=0)
        dVc_dt = np.concatenate(dVc_dts, axis=0)
        Isc = np.concatenate(Iscs, axis=0)

        return dVc_dt, Isc, dt

    def _logp(self, log_gs, dV, Is, dt, prior):
        """
        Compute the log prob of a set of conductances given
        the estimated dV and I
        """
        gs = np.exp(log_gs)
        C = gs.size
        T = dV.size
        dV = dV.reshape((T,1))
        Is = Is.reshape((T,C))
        dt = dt.reshape((T,1))

        sig_V = hypers['sig_V'].value
        pred_dV = np.dot(Is, gs)

        ll = np.sum(-0.5/sig_V**2 * (dV - pred_dV)**2)
        # ll = 0.
        lprior = prior.logp_logx(log_gs)
        # lprior = 0

        lp = ll + lprior
        if np.isnan(lp):
            lp = -np.Inf
        return lp

    def _grad_logp(self, log_gs, dV, Is, dt, prior):
        """
        Compute the gradient of the log prob of a set of conductances
        given the estimated dV and I
        """
        C = log_gs.size
        T = dV.size

        gs = np.exp(log_gs).reshape((C,1))
        dV = dV.reshape((T,1))
        Is = Is.reshape((T,C))
        dt = dt.reshape((T,1))

        sig_V = hypers['sig_V'].value

        preddV = np.dot(Is, gs)
        dpreddV_dg = Is
        dll_dg = np.dot((1.0/sig_V**2 * (dV - preddV)).T, dpreddV_dg)
        dll_dg = dll_dg.reshape((C,1))
        # dll_dg = np.zeros((C,1))
        dlprior_dg = prior.grad_logp_logx_wrt_x(gs)
        dlprior_dg = dlprior_dg.reshape((C,1))
        # dlprior_dg = np.zeros((C,1))

        dlp_dg = dll_dg + dlprior_dg
        dg_dlogg = gs

        dlp_dlogg = dlp_dg * dg_dlogg

        # print "gs: ", gs
        # print "dlp/dlng: ", dlp_dlogg.ravel()
        # if not np.all(np.isfinite(dlp_dlogg)):
        #     import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # import matplotlib.pyplot as plt
        # plt.figure(2)
        # plt.plot(dV, '-k')
        # plt.plot(preddV, '-r')
        # plt.show()
        # # plt.pause(0.1)
        # # # raw_input('Press any key to continue\n')
        # import pdb; pdb.set_trace()

        return dlp_dlogg.reshape((C,))

    def check_grads(self, f, df, xs, step=1e-4):
        for x in xs:
            dx = step*np.random.randn()
            x2 = x + dx
            f2 = f(x2)
            pred_f2 = f(x) + df(x)*dx

            try:
                assert np.allclose(f2, pred_f2)

            except:
                print "x: ", x
                print "f(x): ", f(x)
                print "dx: ", dx
                print "x2: ", x2
                print "df(x): ", df(x)
                print "f2: ", f2
                print "pred_f2: ", pred_f2

                import pdb; pdb.set_trace()

    def joint_update(self, model):
        """
        Sample the conductances of the neuron given its latent state variables
        """
        # c = self.compartment
        c = self.get_compartment(model)
        # Get the sub-structured arrays for this comp
        chs = c.channels

        # Get a list of conductances for this compartment
        gs = [ch.g for ch in chs]
        gs_values = np.array([g.value for g in gs]).ravel()

        dVc_dt, Isc, dt = self.get_dV_and_currents(model)

        # Sample new gs with HMC
        prior = ProductDistribution([g.distribution for g in gs])
        nll = lambda log_gs: -1.0*self._logp(log_gs,dVc_dt, Isc, dt, prior )
        grad_nll = lambda log_gs: -1.0*self._grad_logp(log_gs,dVc_dt, Isc, dt, prior)

        stepsz = 0.005
        nsteps = 10
        log_gs = hmc(nll, grad_nll, stepsz, nsteps, np.log(gs_values))
        gs = np.exp(log_gs)

        # Update the channel conductance parameter
        for g,ch in zip(gs,chs):
            ch.g.value = g

    def serial_update(self, model):
        # Sample each conductance in turn given that
        # C*dVc_dt ~ N(I_in - np.dot(gsc, Isc), sig_V^2)

        c = self.get_compartment(model)
        chs = c.channels

        # Get a list of conductances for this compartment
        gs = [ch.g for ch in chs]
        gsc = np.array([g.value for g in gs]).ravel()
        # import pdb; pdb.set_trace()

        dVc_dt, Isc, dt = self.get_dV_and_currents(model)

        for (i,(g,ch)) in enumerate(zip(gs, chs)):
            i_rem = np.concatenate((np.arange(i), np.arange(i+1,len(chs))))
            gsc_rem = gsc[i_rem]
            Isc_rem = Isc[:,i_rem]
            dV_resid = dVc_dt - np.dot(Isc_rem, gsc_rem).ravel()

            # Sample new gs with HMC
            prior = g.distribution
            nll = lambda log_gs: -1.0*self._logp(log_gs,dV_resid, Isc[:,i], dt, prior )
            grad_nll = lambda log_gs: -1.0*self._grad_logp(log_gs,dV_resid, Isc[:,i], dt, prior)

            # DEBUG:
            # self.check_grads(nll, grad_nll, np.log(gsc[i]).reshape((1,)), step=1e-4)

            nsteps = 10
            curr_log_g = np.log(gsc[i]).reshape((1,))
            new_log_g, new_step_sz, new_accept_rate = \
                hmc(nll, grad_nll, self.step_sz[i], nsteps, curr_log_g,
                    adaptive_step_sz=True,
                    avg_accept_rate=self.avg_accept_rate[i],
                    min_step_sz=1e-4)

            # Update step size and accept rate
            self.step_sz[i] = new_step_sz
            self.avg_accept_rate[i] = new_accept_rate

            # Update the channel conductance parameter
            gsc[i] = np.exp(new_log_g)
            ch.g.value = np.exp(new_log_g)

    def update(self, model):
        return self.serial_update(model)

    def gradient_V(self, t, V):
        """
        Compute the gradient of V
        """
        # Compute dV/dt using first order differences
        dVdt = np.zeros_like(V)
        dVdt[:-1] = (V[1:]-V[:-1])/(t[1:]-t[:-1])
        dVdt[-1] = (V[-1]-V[-2])/(t[-1]-t[-2])
        return dVdt


class NeuronLatentStateUpdate(MetropolisHastingsUpdate):
    """
    Update our estimate of the latent state of the neuron with particle MCMC.
    """

    def __init__(self, model, population):
        self.model = model
        self.population = population

    def preprocess(self):
        """

        :return:
        """
        self.N_particles = hypers['N_particles'].value

        # Set up initial state distribution
        # Initial state is centered around the steady state
        D = sz_dtype(self.population.latent_dtype)
        self.mu_initial = self.population.steady_state().reshape((D,1))

        # TODO: Implement a distribution over the initial variances
        sig_initial = np.ones(1, dtype=self.population.latent_dtype)
        sig_initial.fill(np.asscalar(hypers['sig_ch_init'].value))
        for neuron in self.population.neurons:
            for compartment in neuron.compartments:
                sig_initial[neuron.name][compartment.name]['V'] = hypers['sig_V_init'].value
        self.sig_initial = as_matrix(sig_initial)

        # TODO: Implement a distribution over the  transition noise
        sig_trans = np.ones(1, dtype=self.population.latent_dtype)
        sig_trans.fill(np.asscalar(hypers['sig_ch'].value))
        for neuron in self.population.neurons:
            for compartment in neuron.compartments:
                sig_trans[neuron.name][compartment.name]['V'] = hypers['sig_V'].value
        self.sig_trans = as_matrix(sig_trans)

    def update(self, model, cache=None):
        """

        :param current_state:
        :return:
        """
        population = model.population

        # Update each data sequence one at a time
        for data in model.data_sequences:
            t = data.t
            T = data.T
            latent = data.latent
            inpt = data.input
            state = data.states
            obs = data.observations

            # View the latent state as a matrix
            D = sz_dtype(latent.dtype)
            z = as_matrix(latent, D)
            x = as_matrix(obs)


            lb = population.latent_lb
            ub = population.latent_ub
            p_initial = StaticTruncatedGaussianDistribution(D, self.mu_initial, self.sig_initial, lb, ub)

            # The observation model gives us a noisy version of the voltage
            lkhd = ObservationLikelihood(model)

            # The transition model is a noisy Hodgkin Huxley proposal
            # prop = TruncatedHodgkinHuxleyProposal(population, t, inpt, self.sig_trans)
            prop = HodgkinHuxleyProposal(population, t, inpt, self.sig_trans)

            # Run the particle Gibbs step with ancestor sampling
            # Create a conditional particle filter with ancestor sampling
            pf = ParticleGibbsAncestorSampling(D, T, 0, x[:,0],
                                               prop, lkhd, p_initial, z,
                                               Np=self.N_particles)

            for ind in np.arange(1,T):
                x_curr = x[:,ind]
                pf.filter(ind, x_curr)

            # Sample a particular weight trace given the particle weights at time T
            i = np.random.choice(np.arange(pf.Np), p=pf.trajectory_weights)
            # z_inf = pf.trajectories[:,i,:].reshape((D,T))
            z_inf = pf.get_trajectory(i).reshape((D,T))

            # Update the data sequence's latent and state
            data.latent = as_sarray(z_inf, population.latent_dtype)
            data.states = population.evaluate_state(data.latent, inpt)


class DirectCompartmentVoltageUpdate(MetropolisHastingsUpdate):
    """
    An update for the parameters of the direct compartment voltage
    observation. E.g. the noise level of the observation.
    """
    def __init__(self, model):
        raise NotImplementedError()

    def update(self, model):
        raise NotImplementedError()

class LinearFluorescenceUpdate(MetropolisHastingsUpdate):
    """
    An update for the parameters of the linear fluorescence voltage
    observation.
    """
    def __init__(self, model):
        self.model = model

    def update(self, model):
        for data in self.model.data_sequences:
            for obs in self.model.observation.observations:
                obs.update(data.latent, data.observations)

class SigmaTransitionUpdate(MetropolisHastingsUpdate):
    """
    TODO: Implement a sampler for the transition noise.
    This may make it easier to mix over voltage trajectories
    """
    def update(self, model, cache=None):
        """
        TODO: Handle truncated Gaussians noise for channel params
        """
        # Keep sufficient stats for a Gaussian noise model
        N = 0
        beta_V_hat = 0
        # beta_ch_hat = 0

        # Get the latent voltages
        for ds in model.data_sequences:
            latent = as_matrix(ds.latent)
            # The transition model is a noisy Hodgkin Huxley proposal
            prop = TruncatedHodgkinHuxleyProposal(model.population, ds.t, ds.input,
                                                  as_matrix(np.zeros(1, dtype=model.population.latent_dtype)))
            dt = ds.t[1:] - ds.t[:-1]
            # import pdb; pdb.set_trace()
            pred = np.zeros((latent.shape[0], ds.T-1))
            for t in np.arange(ds.T-1):
                pt,_,_ = prop.sample_next(t, latent[:,t,np.newaxis], t+1)
                pred[:,t] = pt.ravel()

            # pred2,_ = prop.sample_next(np.arange(ds.T-1),
            #                           latent[:,:-1],
            #                           np.arange(1,ds.T))

            # if not np.allclose(pred, pred2):
            #     import pdb; pdb.set_trace()
            for neuron in model.population.neurons:
                for compartment in neuron.compartments:
                    V = get_item_at_path(as_sarray(latent, model.population.latent_dtype),
                                         compartment.path)['V']
                    Vpred = get_item_at_path(as_sarray(pred, model.population.latent_dtype),
                                             compartment.path)['V']
                    dV = (Vpred - V[1:])/dt

                    # Update sufficient stats
                    N += len(dV)
                    beta_V_hat += (dV**2).sum()

        # Sample a new beta_V
        sig2_V = 1.0/np.random.gamma(hypers['a_sig_V'].value + N/2.,
                                     1.0/(hypers['b_sig_V'].value + beta_V_hat/2.))

        hypers['sig_V'].value = np.sqrt(sig2_V)

class GewekeUpdate(MetropolisHastingsUpdate):
    """
    Simple Geweke update
    """
    def update(self, model):
        # import pdb; pdb.set_trace()
        self.t = model.data_sequences[0].t
        self.stim = model.data_sequences[0].stimuli

        from pybiophys.simulation.simulate import simulate

        model.data_sequences[0] = simulate(model, self.t, self.stim)

def initialize_updates(model, geweke=False):
    """
    Take in a model and return a set of MCMC transitions to sample the
    posterior distribution over parameters of the model.

    :param model:
    :return: a list of updates
    """
    # Import the components that we reference
    from pybiophys.observation.observable import NewDirectCompartmentVoltage, LinearFluorescence

    # Make a list of updates
    updates = []

    for neuron in model.population.neurons:
        # TODO: Support updates of individual neuron latent states?

        # Add updates for the compartment conductances
        for compartment in neuron.compartments:
            # TODO: Check if the conductances are parameters?
            # updates.append(ConductanceUpdate(compartment))
            # updates.append(HmcConductanceUpdate(compartment))
            pass

    # Add an update for the latent state of the population
    updates.append(NeuronLatentStateUpdate(model, model.population))

    # Check for a voltage observation
    # for obs in model.observation.observations:
    #     if isinstance(obs, NewDirectCompartmentVoltage):
    #         updates.append(DirectCompartmentVoltageUpdate(model))
    #
    for obs in model.observation.observations:
        if isinstance(obs, LinearFluorescence):
            updates.append(LinearFluorescenceUpdate(model))

    # Update the transition noise
    # updates.append(SigmaTransitionUpdate())

    # Add a Geweke Update
    if geweke:
        updates.append(GewekeUpdate())

    # Preprocess the updates
    for update in updates:
        update.preprocess()

    return updates
