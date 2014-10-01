from model import *
from pybiophys.neuron.neuron          import Neuron
from pybiophys.observation.observable import VectorObserver
from pybiophys.utils                  import utils
from pybiophys.plotting.plotting      import plot_latent_state
import matplotlib.pyplot              as     plt
from scipy.integrate                  import odeint

class Instance(object):
    def __init__(self, model):
        self.neuron   = Neuron(model.description)
        self.observer = VectorObserver(model.description)
        # Maybe this should also keep track of the stimulus?

    @property
    def stimulus_dtype(self):
        return self.neuron.input_dtype

    @property
    def neuron_latent_dtype(self):
        return self.neuron.latent_dtype

    @property
    def observable_dtype(self):
        return self.observer.observed_dtype

    def latent_state_dynamics(self):
        """
        Returns a function that gives you the derivative of the latent state with respect to time
        for a given stimulus (not instance time)
        """
        def _hh_dynamics(state, stim):
            inpt = utils.array_to_dtype(stim, np.dtype(self.stimulus_dtype))
            dzdt = self.neuron.kinetics(state.ravel('F').view(self.neuron_latent_dtype), inpt)
            return dzdt.view(np.float)

        return _hh_dynamics
    
    def steady_state(self):
        return self.neuron.steady_state()
        
    def _latent_stimulate(self, t, stim, z0 = None, intmethod='forward_euler'):
        """
        Copies the ability to stimulate the neuron from the ipython notebook
        """
        if not z0:
            z0 = self.steady_state()

        dynamics = self.latent_state_dynamics()
        def _hh_dynamics(state, ti):
            return dynamics(state, stim[ti])

        if method.lower() == 'forward_euler':
            T = len(t)
            z0 = z0.view(np.float)
            D = len(z0)
            z = np.zeros((D, T))
            z[:, 0] = z0
            lb = self.neuron.latent_lb
            ub = self.neuron.latent_ub

            for ti in np.arange(1, T):
                z[:, ti] = z[:, ti-1] + (t[ti]-t[ti-1]) * _hh_dynamics(z[:, ti-1], ti-1)
                z[:, ti] = np.clip(z[:, ti], lb, ub)

        elif method.lower() == 'odeint':
            z = odeint(_hh_dynamics, z0.view(np.float), t)

        return z

    def stimulate(self, t, stim, z0 = None):
        stim_vec = utils.array_to_dtype(np.array(stim.intensity(t)), np.dtype(self.stimulus_dtype))
        latent_trajectory     = self._latent_stimulate(t, stim_vec, z0)
        typed_latent = utils.array_to_dtype(latent_trajectory, np.dtype(self.neuron_latent_dtype))
        observable_trajectory = self.observer.transform(typed_latent)
        return observable_trajectory, latent_trajectory

    def graph_channel_currents(self):
        for comp in self.neuron.compartments:
            for channel in comp.channels:
                channel.IV_plot()

    def graph_latent_trajectory(self, latent, t):
        D, T     = latent.shape
        fig, axs = plt.subplots(D, 1, figsize=(15, 27))
        plot_latent_state(t, latent, self.neuron_latent_dtype, fig=fig)
        plt.show()
