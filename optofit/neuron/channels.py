
"""
Define the ionic channels used by a neuron
"""
import numpy as np
from numpy import exp

from optofit.models.model import *
from optofit.models.component import Component
from optofit.models.parameters import Parameter
from optofit.models.hyperparameters import hypers
from optofit.inference.distributions import GammaDistribution

from optofit.utils.utils import get_item_at_path
#
# def make_channel(compartment, model):
#     """
#     Make a channel with the given channel model.
#     """
#     if model['channel_type'].lower() == 'leak':
#         return LeakChannel(compartment, model)
#     elif model['channel_type'].lower() == 'na' or \
#          model['channel_type'].lower() == 'sodium':
#         return NaChannel(compartment, model)
#
#     # Hippocampal CA3 style sodium channel
#     elif model['channel_type'].lower() == 'ca3na' or \
#          model['channel_type'].lower() == 'ca3_sodium':
#         return Ca3NaChannel(compartment, model)
#
#     # Delayed rectification is the default potassium channel
#     elif model['channel_type'].lower() == 'kdr' or \
#          model['channel_type'].lower() == 'k' or \
#          model['channel_type'].lower() == 'potassium':
#         return KdrChannel(compartment, model)
#
#     elif model['channel_type'].lower() == 'ca3kdr':
#         return Ca3KdrChannel(compartment, model)
#
#    # Delayed rectification is the default potassium channel
#     elif model['channel_type'].lower() == 'ca3ka':
#         return Ca3KaChannel(compartment, model)
#
#     # Hippocampal CA3 style calcium channel
#     elif model['channel_type'].lower() == 'ca3ca' or \
#          model['channel_type'].lower() == 'ca3_calcium' or \
#          model['channel_type'].lower() == 'calcium':
#         return Ca3CaChannel(compartment, model)
#
#     elif model['channel_type'].lower() == "kahp" or \
#          model['channel_type'].lower() == "ca3kahp":
#         return Ca3KahpChannel(compartment, model)
#
#     elif model['channel_type'].lower() == "ca3kc":
#         return Ca3KcChannel(compartment, model)
#
#     elif model['channel_type'].lower() == "chr2":
#         return ChR2Channel(compartment, model)
#
#     else:
#         raise Exception("Unrecognized channel type: %s" % model['channel_type'])

class Channel(Component):
    """
    Abstract base class for an ion channel.
    """

    def __init__(self, name, compartment):
        super(Channel, self).__init__()
        self.parent = compartment
        self.compartment = self.parent
        self.name = name

        # All channels (at least so far!) have a conductance and a reversal
        # potential
        self.g = None
        self.E = None

        self._latent_dtype = []
        self._state_dtype = []
        self._input_dtype = None
        self._latent_lb = []
        self._latent_ub = []

        self._moves_calcium = False
        self._calcium_dependent = False
        
    @property
    def moves_calcium(self):
        return self._moves_calcium

    @property
    def calcium_dependent(self):
        return self._calcium_dependent

    @property
    def latent_dtype(self):
        return self._latent_dtype

    @latent_dtype.setter
    def latent_dtype(self, value):
        self._latent_dtype = value

    @property
    def state_dtype(self):
        return self._state_dtype

    @state_dtype.setter
    def state_dtype(self, value):
        self._state_dtype = value

    @property
    def input_dtype(self):
        return self._input_dtype
    @input_dtype.setter
    def input_dtype(self, value):
        self._input_dtype = value
    # Add properties for constraints on the latent variables
    @property
    def latent_lb(self):
        return self._latent_lb

    @latent_lb.setter
    def latent_lb(self, value):
        self._latent_lb = value

    @property
    def latent_ub(self):
        return self._latent_ub

    @latent_ub.setter
    def latent_ub(self, value):
        self._latent_ub = value

    def steady_state(self, V):
        # Steady state value of the latent vars as a function of voltage
        return np.array([])

    def kinetics(self, latent, inpt, state):
        pass

    def IV_plot(self, start=-200, stop=100):
        comp_state_dt = np.dtype(self.compartment._state_vars)

        if self.latent_dtype:
            dt = np.dtype([(self.compartment.name, [('V', np.float64), ('[Ca]', np.float64), (self.name, self.latent_dtype)])])
        else:
            dt = np.dtype([(self.compartment.name, [('V', np.float64), ('[Ca]', np.float64)])])

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm

        if self.calcium_dependent:
            Vs  = np.linspace(start, stop, 100)
            Cas = np.linspace(0, 1000, 100)
            
            X, Y = np.meshgrid(Vs, Cas)
            
            Z = np.zeros(X.shape)
            for row in range(X.shape[0]):
                for col in range(X.shape[1]):
                    state  = np.ndarray(buffer = np.array([X[row, col], Y[row, col]]), dtype=comp_state_dt, shape = comp_state_dt.shape)
                    latent = np.ndarray(buffer = np.hstack((np.array([X[row, col], Y[row, col]]), self.steady_state(state))), dtype = dt, shape = dt.shape)
                    Z[row, col] = self.evaluate_state(np.array([latent]), ())[0][0]
                
            fig  = plt.figure()
            ax   = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
            plt.title(self.name)
            plt.show()
        else:
            ca = 0
            Vs = np.linspace(start, stop, 1000)
            
            state = np.ndarray(buffer = np.array([Vs[0], ca]), dtype=comp_state_dt, shape = comp_state_dt.shape)
            latents = np.ndarray(buffer=np.hstack((np.array([Vs[0], ca]), self.steady_state(state))), dtype = dt, shape = dt.shape)

            for v in Vs[1:]:
                state = np.ndarray(buffer = np.array([v, ca]), dtype=comp_state_dt, shape = comp_state_dt.shape)
                latents = np.append(latents, np.ndarray(buffer=np.hstack((np.array([v, ca]), self.steady_state(state))), dtype = dt, shape = dt.shape))
    
            plt.plot(Vs, [self.evaluate_state(l, ()) for l in latents])
            plt.title(self.name)
            plt.show()

    def _set_defaults(self, g, g_param, E, E_param):
        if g is None:
            self.g = g_param
        else:
            self.g = g

        if E is None:
            self.E = E_param
        else:
            self.E = E

class LeakChannel(Channel):
    """
    Passive leak channel.
    """
    def __init__(self, name, compartment,
                 g_leak=None, E_leak=None):
        super(LeakChannel, self).__init__(name, compartment)
        self.state_dtype = [('I', np.float64)]


        # By default, g is gamma distributed
        if g_leak is None:
            self.g = Parameter('g_leak',
                     distribution=GammaDistribution(hypers['a_g_leak'].value,
                                                    hypers['b_g_leak'].value),
                     lb=0.0)

        else:
            assert isinstance(g_leak, Parameter)
            self.g = g_leak

        # By default, E is a hyperparameter
        if E_leak is None:
            self.E = hypers['E_leak']
        else:
            assert isinstance(E_leak, Parameter)
            self.E = E_leak


    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of this compartment
        """
        state = np.zeros(latent.shape, dtype=self.state_dtype)
        x_comp = get_item_at_path(latent, self.compartment.path)
        state['I'] = x_comp['V'] - self.E.value
        return state

    def kinetics(self, latent, inpt, state):
        """
        Compute the state kinetics, d{latent}/dt, according to the Hodgkin-Huxley eqns,
        given current state x and external or given variables y.

        latent:  latent state variables of the neuron, e.g. voltage per compartment,
                 channel activation variables, etc.

        inpt:    observations, e.g. supplied irradiance, calcium concentration, injected
                 current, etc.

        state:   evaluated state of the neuron including channel currents, etc.

        returns:
        dxdt:   Rate of change of the latent state variables.
        """
        return None

class NaChannel(Channel):
    """
    Sodium channel.
    """
    def __init__(self, name, compartment,
                 g_na=None, E_na=None):
        super(NaChannel, self).__init__(name, compartment)
        self.latent_dtype = [('m', np.float64), ('h', np.float64)]
        self.latent_lb = np.array([0,0])
        self.latent_ub = np.array([1,1])
        self.state_dtype = [('I', np.float64)]

        # By default, g is gamma distributed
        if g_na is None:
            self.g = Parameter('g_na',
                     distribution=GammaDistribution(hypers['a_g_na'].value,
                                                    hypers['b_g_na'].value),
                     lb=0.0)

        else:
            self.g = g_na

        # By default, E is a hyperparameter
        if E_na is None:
            self.E = hypers['E_Na']
        else:
            self.E = E_na

    def steady_state(self, state):
        V = state['V']
        # Steady state value of the latent vars
        # Compute the alpha and beta as a function of V
        am1 = 0.1*(V+35.)/(1-exp(-(V+35.)/10.))
        ah1 = 0.07*exp(-(V+50.)/20.)
        bm1 = 4.*exp(-(V+65.)/18.)
        bh1 = 1./(exp(-(V+35)/10.)+1)

        xss = np.zeros(2)
        xss[0] = am1/(am1+bm1)
        xss[1] = ah1/(ah1+bh1)
        return xss

    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of this compartment
        """
        state = np.zeros(latent.shape, dtype=self.state_dtype)
        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)
        state['I'] = x_ch['m']**3 * x_ch['h'] * (x_comp['V'] - self.E.value)
        return state

    def kinetics(self, latent, inpt, state):
        """
        Compute the state kinetics, d{latent}/dt, according to the Hodgkin-Huxley eqns,
        given current state x and external or given variables y.

        latent:  latent state variables of the neuron, e.g. voltage per compartment,
                 channel activation variables, etc.

        inpt:    observations, e.g. supplied irradiance, calcium concentration, injected
                 current, etc.

        state:   evaluated state of the neuron including channel currents, etc.

        returns:
        dxdt:   Rate of change of the latent state variables.
        """
        # Initialize dxdt for each latent state
        dxdt = np.zeros(latent.shape, dtype=self.latent_dtype)

        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)

        V = x_comp['V']
        m = x_ch['m']
        h = x_ch['h']

        # Compute the alpha and beta as a function of V
        am1 = 0.1*(V+35.)/(1-exp(-(V+35.)/10.))
        ah1 = 0.07*exp(-(V+50.)/20.)

        bm1 = 4.*exp(-(V+65.)/18.)
        bh1 = 1./(exp(-(V+35.)/10.)+1.)

        # Compute the channel state updates
        dxdt['m'] = am1*(1.-m) - bm1*m
        dxdt['h'] = ah1*(1.-h) - bh1*h

        return dxdt

class Ca3NaChannel(Channel):
    """
    Sodium channel in a hippocampal CA3 neuron.
    """
    def __init__(self, name, compartment,
                 g_ca3na = None,
                 E_ca3na = None):

        super(Ca3NaChannel, self).__init__(name, compartment)
        self._latent_dtype = [('m', np.float64), ('h', np.float64)]
        self._latent_lb = np.array([0,0])
        self._latent_ub = np.array([1,1])
        self._state_dtype = [('I', np.float64)]
        self._input_dtype = None
        
        self._set_defaults(g_ca3na, Parameter('g_ca3na', distribution=
                                              GammaDistribution(
                                                  hypers['a_g_ca3na'].value,
                                                  hypers['b_g_ca3na'].value
                                              ),
                                              lb=0.0),
                           E_ca3na,  hypers['E_Na'])

    @property
    def latent_dtype(self):
        return self._latent_dtype

    @property
    def state_dtype(self):
        return self._state_dtype

    @property
    def input_dtype(self):
        return self._input_dtype

    @property
    def latent_lb(self):
        return self._latent_lb

    @property
    def latent_ub(self):
        return self._latent_ub

    def steady_state(self, state):
        V = state['V']
        # Steady state value of the latent vars
        # Compute the alpha and beta as a function of V
        V_ref = V + 60
        am1 = 0.32*(13.1-V_ref)/(exp((13.1-V_ref)/4)-1)
        ah1 = 0.128*exp((17.0-V_ref)/18.0)

        bm1 = 0.28*(V_ref-40.1)/(exp((V_ref-40.1)/5.0)-1.0)
        bh1 = 4.0/(1.0+exp((40.-V_ref)/5.0))

        xss = np.zeros(2)
        xss[0] = am1/(am1+bm1)
        xss[1] = ah1/(ah1+bh1)
        return xss

    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of this compartment
        """
        state = np.zeros(latent.shape, dtype=self.state_dtype)
        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)
        state['I'] = x_ch['m']**2 * x_ch['h'] * (x_comp['V'] - self.E.value)
        return state

    def kinetics(self, latent, inpt, state):
        """
        Compute the state kinetics, d{latent}/dt, according to the Hodgkin-Huxley eqns,
        given current state x and external or given variables y.

        latent:  latent state variables of the neuron, e.g. voltage per compartment,
                 channel activation variables, etc.

        inpt:    observations, e.g. supplied irradiance, calcium concentration, injected
                 current, etc.

        state:   evaluated state of the neuron including channel currents, etc.

        returns:
        dxdt:   Rate of change of the latent state variables.
        """
        # Initialize dxdt for each latent state
        dxdt = np.zeros(latent.shape, dtype=self.latent_dtype)

        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)

        V = x_comp['V']
        # Use resting potential of zero
        V_ref = V + 60

        m = x_ch['m']
        h = x_ch['h']

        # Compute the alpha and beta as a function of V
        am1 = 0.32*(13.1-V_ref)/(exp((13.1-V_ref)/4)-1)
        ah1 = 0.128*exp((17.0-V_ref)/18.0)

        bm1 = 0.28*(V_ref-40.1)/(exp((V_ref-40.1)/5.0)-1.0)
        bh1 = 4.0/(1.0+exp((40.-V_ref)/5.0))

        # Compute the channel state updates
        dxdt['m'] = am1*(1.-m) - bm1*m
        dxdt['h'] = ah1*(1.-h) - bh1*h

        return dxdt


class KdrChannel(Channel):
    """
    Potassium (delayed rectification) channel.
    """
    def __init__(self, name, compartment,
                 g_kdr=None, E_kdr=None):
        super(KdrChannel, self).__init__(name, compartment)
        self.latent_dtype = [('n', np.float64)]
        self.state_dtype = [('I', np.float64)]
        self.latent_lb = np.array([0])
        self.latent_ub = np.array([1])

        # By default, g is gamma distributed
        if g_kdr is None:
            self.g = Parameter('g_kdr',
                     distribution=GammaDistribution(hypers['a_g_kdr'].value,
                                                    hypers['b_g_kdr'].value),
                     lb=0.0)

        else:
            self.g = g_kdr

        # By default, E is a hyperparameter
        if E_kdr is None:
            self.E = hypers['E_K']
        else:
            self.E = E_kdr

    def steady_state(self, state):
        # Steady state activation values
        V  = state['V'] + 60
        an1 = 0.01*(V+55.)/(1-exp(-(V+55.)/10.))
        bn1 = 0.125*exp(-(V+65.)/80.)

        xss = np.zeros(1)
        xss[0] = an1/(an1+bn1)
        return xss

    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of this compartment
        """
        state = np.zeros(latent.shape, dtype=self.state_dtype)
        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)
        state['I'] = x_ch['n']**4 * (x_comp['V'] - self.E.value)
        return state

    def kinetics(self, latent, inpt, state):
        """
        Compute the state kinetics, d{latent}/dt, according to the Hodgkin-Huxley eqns,
        given current state x and external or given variables y.

        latent:  latent state variables of the neuron, e.g. voltage per compartment,
                 channel activation variables, etc.

        inpt:    observations, e.g. supplied irradiance, calcium concentration, injected
                 current, etc.

        state:   evaluated state of the neuron including channel currents, etc.

        returns:
        dxdt:   Rate of change of the latent state variables.
        """
        # Initialize dxdt for each latent state
        dxdt = np.zeros(latent.shape, dtype=self.latent_dtype)

        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)

        V = x_comp['V'] + 60
        n = x_ch['n']

        # Compute the alpha and beta as a function of V
        an1 = 0.01*(V+55.) /(1-exp(-(V+55.)/10.))
        bn1 = 0.125*exp(-(V+65.)/80.)

        # Compute the channel state updates
        dxdt['n'] = an1 * (1.0-n) - bn1*n

        return dxdt

class Ca3KdrChannel(Channel):
    """
    Potassium (delayed rectification) channel from Traub.
    """
    def __init__(self, name, compartment,
                 g_ca3kdr = None,
                 E_ca3kdr = None):

        super(Ca3KdrChannel, self).__init__(name, compartment)
        self._latent_dtype = [('n', np.float64)]
        self._state_dtype = [('I', np.float64)]
        self._input_dtype = None
        self._latent_lb = np.array([0])
        self._latent_ub = np.array([1])

        self._set_defaults(g_ca3kdr, Parameter('g_ca3kdr', distribution=
                                              GammaDistribution(
                                                  hypers['a_g_ca3kdr'].value,
                                                  hypers['b_g_ca3kdr'].value
                                              ),
                                              lb=0.0),
                           E_ca3kdr,  hypers['E_K'])

    @property
    def latent_dtype(self):
        return self._latent_dtype

    @property
    def state_dtype(self):
        return self._state_dtype

    @property
    def input_dtype(self):
        return self._input_dtype

    @property
    def latent_lb(self):
        return self._latent_lb

    @property
    def latent_ub(self):
        return self._latent_ub


    def alpha_beta(self, state):
        V = state['V'] + 60
        # Traub 1991
        alpha = .016*(35.1 - V) / (np.exp((35.1-V)/5)-1)
        beta  = .25 * np.exp((20 - V)/40)

        """
        # Traub 1993
        alpha = .03 * (17.2 - V) / (np.exp((17.2 -V) / 5) - 1)
        beta  = .45 * np.exp((12 - V) / 40)
        """
        return alpha, beta

    def steady_state(self, state):
        # Steady state activation values
        alpha, beta = self.alpha_beta(state)

        xss = np.zeros(1)
        xss[0] = alpha/(alpha+beta)
        return xss

    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of this compartment
        """
        state = np.zeros(latent.shape, dtype=self.state_dtype)
        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)
        """
        # Traub 1993
        state['I'] = x_ch['n']**2 * (x_comp['V'] - self.E.value)
        """

        # Traub 1991 
        state['I'] = x_ch['n']**4 * (x_comp['V'] - self.E.value)
        return state

    def kinetics(self, latent, inpt, state):
        """
        Compute the state kinetics, d{latent}/dt, according to the Hodgkin-Huxley eqns,
        given current state x and external or given variables y.

        latent:  latent state variables of the neuron, e.g. voltage per compartment,
                 channel activation variables, etc.

        inpt:    observations, e.g. supplied irradiance, calcium concentration, injected
                 current, etc.

        state:   evaluated state of the neuron including channel currents, etc.

        returns:
        dxdt:   Rate of change of the latent state variables.
        """
        # Initialize dxdt for each latent state
        dxdt = np.zeros(latent.shape, dtype=self.latent_dtype)

        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)
        n = x_ch['n']

        # Compute the alpha and beta as a function of V
        alpha, beta = self.alpha_beta(x_comp)

        # Compute the channel state updates
        dxdt['n'] = alpha * (1.0-n) - beta*n

        return dxdt


class Ca3KahpChannel(Channel):
    """
    Potassium (after hyperpolarization) channel.
    """
    def __init__(self, name, compartment,
                 g_ca3kahp = None,
                 E_ca3kahp = None):
        super(Ca3KahpChannel, self).__init__(name, compartment)

        # Kahp requires Calcium compartment
        from compartment import CalciumCompartment
        # TODO: Or observed calcium?
        assert isinstance(compartment, CalciumCompartment)

        self._latent_dtype = [('q', np.float64)]
        self._state_dtype = [('I', np.float64)]
        self._input_dtype = None
        self._latent_lb = np.array([0])
        self._latent_ub = np.array([1])

        self._calcium_dependent = True

        self._set_defaults(g_ca3kahp, Parameter('g_ca3kahp', distribution=
                                              GammaDistribution(
                                                  hypers['a_g_ca3kahp'].value,
                                                  hypers['b_g_ca3kahp'].value
                                              ),
                                              lb=0.0),
                           E_ca3kahp,  hypers['E_Kahp'])

    @property
    def calcium_dependent(self):
        return self._calcium_dependent

    @property
    def latent_dtype(self):
        return self._latent_dtype

    @property
    def state_dtype(self):
        return self._state_dtype

    @property
    def input_dtype(self):
        return self._input_dtype

    @property
    def latent_lb(self):
        return self._latent_lb

    @property
    def latent_ub(self):
        return self._latent_ub

    def steady_state(self, state):
        c_Ca = state['[Ca]']
        #
        # q = X(1,:);
        #
        # % Offset to resting potential of 0
        
        #
        # % Compute the alpha and beta as a function of V
        aq1 = np.min((0.2e-4)*c_Ca, 0.01)
        bq1 = 0.001
        
        return np.array([aq1/(aq1 + bq1)])

    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of this compartment
        """
        state = np.zeros(latent.shape, dtype=self.state_dtype)
        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)
        state['I'] = x_ch['q'] * (x_comp['V'] - self.E.value)
        return state

    def kinetics(self, latent, inpt, state):
        """
        Compute the state kinetics, d{latent}/dt, according to the Hodgkin-Huxley eqns,
        given current state x and external or given variables y.

        latent:  latent state variables of the neuron, e.g. voltage per compartment,
                 channel activation variables, etc.

        inpt:    observations, e.g. supplied irradiance, calcium concentration, injected
                 current, etc.

        state:   evaluated state of the neuron including channel currents, etc.

        returns:
        dxdt:   Rate of change of the latent state variables.
        """
        # Initialize dxdt for each latent state
        dxdt = np.zeros(latent.shape, dtype=self.latent_dtype)

        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)

        V = x_comp['V']
        c_Ca = x_comp['[Ca]']
        q    = x_ch['q']

        # Offset to resting potential of 0
        V_ref = 60 + V

        # % Compute the alpha and beta as a function of V
        aq1 = np.min((0.2e-4)*c_Ca, 0.01)
        bq1 = 0.001
        
        # % Compute the channel state updates
        dxdt['q'] = aq1*(1-q) - bq1*q

        return dxdt



class Ca3KaChannel(Channel):
    """
    Potassium (A-type transient current) channel.
    """
    def __init__(self, name, compartment,
                 g_ca3ka = None,
                 E_ca3ka = None):

        super(Ca3KaChannel, self).__init__(name, compartment)
        self._latent_dtype = [('a', np.float64), ('b', np.float64)]
        self._state_dtype = [('I', np.float64)]
        self._input_dtype = None
        self._latent_lb = np.array([0, 0])
        self._latent_ub = np.array([1, 1])

        self._set_defaults(g_ca3ka, Parameter('g_ca3ka', distribution=
                                              GammaDistribution(
                                                  hypers['a_g_ca3ka'].value,
                                                  hypers['b_g_ca3ka'].value
                                              ),
                                              lb=0.0),
                           E_ca3ka,  hypers['E_K'])


    @property
    def latent_dtype(self):
        return self._latent_dtype

    @property
    def state_dtype(self):
        return self._state_dtype

    @property
    def input_dtype(self):
        return self._input_dtype

    @property
    def latent_lb(self):
        return self._latent_lb

    @property
    def latent_ub(self):
        return self._latent_ub

    def steady_state(self, state):
        V = state['V']
        # Steady state activation values
        # TODO
        xss = np.zeros(2)
        # Offset to resting potential of 0
        V_ref = 60 + V

        # Compute the alpha and beta as a function of V
        aa1 = 0.02*(13.1-V_ref)/(exp((13.1-V_ref)/10.)-1)
        ba1 = 0.0175*(V_ref-40.1)/(exp((V_ref-40.1)/10.) - 1)

        # Inactivation variable b
        ab1 = 0.0016*exp((-13-V_ref)/18.0)
        bb1 = 0.05/(1+exp((10.1-V_ref)/5.0))

        xss[0] = aa1/(aa1+ba1)
        xss[1] = ab1/(ab1+bb1)
        return xss

    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of this compartment
        """
        state = np.zeros(latent.shape, dtype=self.state_dtype)
        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)
        state['I'] = x_ch['a'] * x_ch['b'] * (x_comp['V'] - self.E.value)
        return state

    def kinetics(self, latent, inpt, state):
        """
        Compute the state kinetics, d{latent}/dt, according to the Hodgkin-Huxley eqns,
        given current state x and external or given variables y.

        latent:  latent state variables of the neuron, e.g. voltage per compartment,
                 channel activation variables, etc.

        inpt:    observations, e.g. supplied irradiance, calcium concentration, injected
                 current, etc.

        state:   evaluated state of the neuron including channel currents, etc.

        returns:
        dxdt:   Rate of change of the latent state variables.
        """
        # Initialize dxdt for each latent state
        dxdt = np.zeros(latent.shape, dtype=self.latent_dtype)

        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)

        V = x_comp['V']
        # Offset to resting potential of 0
        V_ref = 60 + V

        # Compute the alpha and beta as a function of V
        aa1 = 0.02*(13.1-V_ref)/(exp((13.1-V_ref)/10.)-1)
        ba1 = 0.0175*(V_ref-40.1)/(exp((V_ref-40.1)/10.)-1)

        # Inactivation variable b
        ab1 = 0.0016*exp((-13.0-V_ref)/18.0)
        bb1 = 0.05/(1+exp((10.1-V_ref)/5.0))

        # Compute the channel state updates
        dxdt['a'] = aa1*(1-x_ch['a']) - ba1*x_ch['a']
        dxdt['b'] = ab1*(1-x_ch['b']) - bb1*x_ch['b']

        return dxdt

class Ca3CaChannel(Channel):
    """
    High Threshold Calcium channel from Traub 1994
    """
    def __init__(self, name, compartment,
                 g_ca3ca = None,
                 E_ca3ca = None):
        super(Ca3CaChannel, self).__init__(name, compartment)
        self._latent_dtype = [('s', np.float64), ('r', np.float64)]
        self._state_dtype = [('I', np.float64)]
        self._input_dtype = None
        self._latent_lb = np.array([0, 0])
        self._latent_ub = np.array([1, 1])
        
        self._moves_calcium = True

        self._set_defaults(g_ca3ca, Parameter('g_ca3ca', distribution=
                                              GammaDistribution(
                                                  hypers['a_g_ca3ca'].value,
                                                  hypers['b_g_ca3ca'].value
                                              ),
                                              lb=0.0),
                           E_ca3ca,  hypers['E_Ca'])

    @property
    def moves_calcium(self):
        return self._moves_calcium

    @property
    def latent_dtype(self):
        return self._latent_dtype

    @property
    def state_dtype(self):
        return self._state_dtype

    @property
    def input_dtype(self):
        return self._input_dtype

    @property
    def latent_lb(self):
        return self._latent_lb

    @property
    def latent_ub(self):
        return self._latent_ub

    def steady_state(self, state):
        V = state['V']
        # Steady state activation values
        V_ref = 60 + V
        alpha = 1.6 / (1 + np.exp(-.072 * (V_ref - 65)))
        beta  = .02 * (V_ref - 51.1) / (np.exp((V_ref - 51.1) / 5) - 1)
        if V_ref <= 0:
            r_alpha = .005
            r_beta  = 0
        else:
            r_alpha = np.exp(-V_ref / 20) / 200
            r_beta = 0.005 - r_alpha
        return np.array([(alpha / (alpha + beta))[0], r_alpha/(r_alpha + r_beta)])

    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of this compartment
        """
        state = np.zeros(len(latent), dtype=self.state_dtype)
        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)
        state['I'] = (x_ch['s'] ** 2) * x_ch['r'] * (x_comp['V'] - self.E.value)
        #print "x_ch: ", x_ch['s']
        return state

    def kinetics(self, latent, inpt, state):
        """
        Compute the state kinetics, d{latent}/dt, according to the Hodgkin-Huxley eqns,
        given current state x and external or given variables y.

        latent:  latent state variables of the neuron, e.g. voltage per compartment,
                 channel activation variables, etc.

        inpt:    observations, e.g. supplied irradiance, calcium concentration, injected
                 current, etc.

        state:   evaluated state of the neuron including channel currents, etc.

        returns:
        dxdt:   Rate of change of the latent state variables.
        """
        # Initialize dxdt for each latent state
        dxdt = np.zeros(latent.shape, dtype=self.latent_dtype)

        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)

        V = x_comp['V']
        # Offset to resting potential of 0
        V_ref = 60 + V

        alpha = 1.6 / (1 + np.exp(-.072 * (V_ref - 65)))
        beta  = .02 * (V_ref - 51.1) / (np.exp((V_ref - 51.1) / 5) - 1)

        r_alpha = np.exp(-V_ref / 20) / 200
        r_alpha[V_ref <= 0] = .005
        r_beta = 0.005 - r_alpha

        dxdt['s'] = alpha * (1 - x_ch['s']) - beta * x_ch['s']
        dxdt['r'] = r_alpha * (1 - x_ch['r']) - r_beta * x_ch['r']
        return dxdt

class Ca3KcChannel(Channel):
    """
    High Threshold Calcium channel from Traub 1994
    """
    def __init__(self, name, compartment,
                 g_ca3kc = None,
                 E_ca3kc = None):

        super(Ca3KcChannel, self).__init__(name, compartment)
        self._latent_dtype = [('c', np.float64)]
        self._state_dtype = [('I', np.float64)]
        self._input_dtype = None
        self._latent_lb = np.array([0])
        self._latent_ub = np.array([1])
        self._calcium_dependent = True

        self._set_defaults(g_ca3kc, Parameter('g_ca3kc', distribution=
                                              GammaDistribution(
                                                  hypers['a_g_ca3kc'].value,
                                                  hypers['b_g_ca3kc'].value
                                              ),
                                              lb=0.0),
                           E_ca3kc,  hypers['E_Ca3Kc'])


    @property
    def moves_calcium(self):
        return self._moves_calcium

    @property
    def calcium_dependent(self):
        return self._calcium_dependent

    @property
    def latent_dtype(self):
        return self._latent_dtype

    @property
    def state_dtype(self):
        return self._state_dtype

    @property
    def input_dtype(self):
        return self._input_dtype

    @property
    def latent_lb(self):
        return self._latent_lb

    @property
    def latent_ub(self):
        return self._latent_ub

    def alpha_beta(self, state):
        V = state['V']
        V_ref = V + 60
        
        alpha = np.zeros(V_ref.shape)
        beta  = np.zeros(V_ref.shape)

        if V_ref.size == 1:
            if V_ref <= 50:
                alpha = (np.exp(((V_ref - 10)/11) - ((V_ref - 6.5)/27)) / 18.975)[V_ref <= 50]
                beta = (2 * np.exp(-1 * (V_ref - 6.5) / 27) - alpha)
            else:
                alpha = 2 * np.exp(-1 * (V_ref - 6.5) / 27)
                beta = 0
        else:
            # Condition 1: V_ref <= 50
            alpha[V_ref<=50] = np.exp(((V_ref[V_ref<=50] - 10)/11) - ((V_ref[V_ref<=50] - 6.5)/27)) / 18.975
            beta[V_ref<=50] = 2 * np.exp(-1 * (V_ref[V_ref<=50] - 6.5) / 27) - alpha[V_ref<=50]
            # Condition 2: V_ref > 50
            alpha[V_ref>50] = 2 * np.exp(-1 * (V_ref[V_ref>50] - 6.5) / 27)
            beta[V_ref>50] = 0.0

            # if V_ref <= 50:
            #     alpha = np.exp(((V_ref - 10)/11) - ((V_ref - 6.5)/27)) / 18.975
            # else:
            #     alpha = 2 * np.exp(-1 * (V_ref - 6.5) / 27)
        
        # beta  = (2 * np.exp(-1 * (V_ref - 6.5) / 27) - alpha)
        return alpha, beta

    def steady_state(self, state):
        alpha, beta = self.alpha_beta(state)
        
        # Steady state activation values
        return np.array(alpha / (alpha + beta))

    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of this compartment
        """
        state = np.zeros(latent.shape, dtype=self.state_dtype)
        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)
        #print "c: ", x_ch['c']
        #print "Ca: ", x_comp['[Ca]'] / 250
        #print "min: ", np.min(1, x_comp['[Ca]'] / 250)
        #print "ans: ", x_ch['c'] * np.min(1, x_comp['[Ca]'] / 250) * (x_comp['V'] - self.E.value)
        state['I'] = x_ch['c'] * np.minimum(1, x_comp['[Ca]'] / 250) * (x_comp['V'] - self.E.value)
        return state

    def kinetics(self, latent, inpt, state):
        """
        Compute the state kinetics, d{latent}/dt, according to the Hodgkin-Huxley eqns,
        given current state x and external or given variables y.

        latent:  latent state variables of the neuron, e.g. voltage per compartment,
                 channel activation variables, etc.

        inpt:    observations, e.g. supplied irradiance, calcium concentration, injected
                 current, etc.

        state:   evaluated state of the neuron including channel currents, etc.

        returns:
        dxdt:   Rate of change of the latent state variables.
        """
        # Initialize dxdt for each latent state
        dxdt = np.zeros(latent.shape, dtype=self.latent_dtype)

        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)

        alpha, beta = self.alpha_beta(x_comp)
        dxdt['c'] = alpha * (1 - x_ch['c']) - beta * x_ch['c']
        return dxdt

class ChR2Channel(Channel):
    """
    Voltage and light gated ChR2 from Williams
    """
    def __init__(self, name, compartment,
                 g_chr2 = None,
                 E_chr2 = None):
        super(ChR2Channel, self).__init__(name, compartment)
        self._latent_dtype = [('O1', np.float64),
                              ('O2', np.float64),
                              ('C1', np.float64),
                              ('C2', np.float64),
                              ('p',  np.float64)]
        self._state_dtype = [('I', np.float64)]
        # self._input_dtype = []
        self._latent_lb = np.array([0, 0, 0, 0, 0])
        self._latent_ub = np.array([1, 1, 1, 1, 1])

        self._set_defaults(g_chr2, Parameter('g_chr2', distribution=
                                              GammaDistribution(
                                                  hypers['a_g_chr2'].value,
                                                  hypers['b_g_chr2'].value
                                              ),
                                              lb=0.0),
                           E_chr2,  hypers['E_ChR2'])

    @property
    def latent_dtype(self):
        return self._latent_dtype

    @property
    def state_dtype(self):
        return self._state_dtype

    @property
    def input_dtype(self):
        return self._input_dtype

    @property
    def latent_lb(self):
        return self._latent_lb

    @property
    def latent_ub(self):
        return self._latent_ub

    def steady_state(self, state):
        ans    = np.zeros(5)

        # start in the closed state
        ans[2] = 0.99
        # ans[3] = 0.99
        
        # Steady state activation values
        return ans

    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of this compartment
        """
        state = np.zeros(latent.shape, dtype=self.state_dtype)
        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)

        # Alert: Is this true?
        V = x_comp['V']

        G   = (10.6408 - 14.6408*np.exp(-V/42.7671)) / V
        gam = 0.1

        state['I'] = G * (x_ch['O1'] + gam*x_ch['O2']) * (x_comp['V'] - self.E.value)
        return state

    def kinetics(self, latent, inpt, state):
        """
        Compute the state kinetics, d{latent}/dt, according to the Hodgkin-Huxley eqns,
        given current state x and external or given variables y.

        latent:  latent state variables of the neuron, e.g. voltage per compartment,
                 channel activation variables, etc.

        inpt:    observations, e.g. supplied irradiance, calcium concentration, injected
                 current, etc.

        state:   evaluated state of the neuron including channel currents, etc.

        returns:
        dxdt:   Rate of change of the latent state variables.
        """
        # import pdb; pdb.set_trace()
        # Initialize dxdt for each latent state
        dxdt = np.zeros(latent.shape, dtype=self.latent_dtype)

        i_comp = get_item_at_path(inpt, self.compartment.path)
        x_comp = get_item_at_path(latent, self.compartment.path)
        x_ch = get_item_at_path(latent, self.path)

        I = i_comp['Irr']
        V = x_comp['V']
        p = x_ch['p']

        # Compute the voltage-sensitive rate constants for state transitions
        Gd1 = 0.075 + 0.043*np.tanh(-(V+20)/20)
        Gd2 = 0.05
        # Gr  = 4.34587e5 * np.exp(-0.0211539274*V)
        Gr = 0.001

        # Define a state variable for time and irradiance dependent activation
        # function for ChR2 (post isomerization)
        theta = 100*I          # Optical stimulation protocol
        tau_chr2 = 1.3         # Time constant for ChR2
        S0   =  0.5*(1+np.tanh(120*(theta-0.1)))
        dxdt['p'] = (S0-p)/tau_chr2

        # Define the light-sensitive rate constants for state transitions
        lamda = 470            # Wavelength of max absorption for retinal
        eps1 = 0.8535          # quantum efficiency for photon absorption from C1
        eps2 = 0.14            # quantum efficiency for photon absorption from C2
        w_loss = 0.77;
        F = 0.00006*I*lamda/w_loss   #Photon flux (molecules/photon/sec)
        # F = (sig_ret/hc)*I*lamda/w_loss*1e-9;   % Photon flux (molecules/photon/sec)

        # Light sensitive rates for C1->01 and C2->O2 transition
        k1 = eps1 * F * p
        k2 = eps2 * F * p

        # Light sensitive O1->02 transitions
        e12d = 0.011
        e12c1 = 0.005
        e12c2 = 0.024
        e12 = e12d + e12c1*np.log(1+I/e12c2)
        # Light sensitive O2->O1 transitions
        e21d = 0.008
        e21c1 = 0.004
        e21c2 = 0.024
        e21 = e21d + e21c1*np.log(1+I/e21c2)

        dxdt['O1'] = k1 * x_ch['C1'] - (Gd1 + e12) * x_ch['O1'] + e21 * x_ch['O2'] 
        dxdt['O2'] = k2 * x_ch['C2'] - (Gd2 + e21) * x_ch['O2'] + e12 * x_ch['O1']
        dxdt['C1'] = Gr * x_ch['C2'] + Gd1 * x_ch['O1'] - k1 * x_ch['C1']
        dxdt['C2'] = Gd2 * x_ch['O2'] + (k2 + Gr) * x_ch['C2']

        return dxdt

    def stationary(self, Irr, V):
        I = Irr
        V = V
        
        dt = np.dtype(self.latent_dtype)
        ans = np.zeros(dt.shape, dtype=dt)

        # Compute the voltage-sensitive rate constants for state transitions
        Gd1 = 0.075 + 0.043*np.tanh(-(V+20)/20)
        Gd2 = 0.05
        Gr  = 4.34587e5 * np.exp(-0.0211539274*V)

        # Define a state variable for time and irradiance dependent activation
        # function for ChR2 (post isomerization)
        theta = 100*I          # Optical stimulation protocol
        S0   =  0.5*(1+np.tanh(120*(theta-0.1)))
        ans['p'] = S0

        # Define the light-sensitive rate constants for state transitions
        lamda = 470            # Wavelength of max absorption for retinal
        eps1 = 0.8535          # quantum efficiency for photon absorption from C1
        eps2 = 0.14            # quantum efficiency for photon absorption from C2
        w_loss = 0.77;
        F = 0.00006*I*lamda/w_loss   #Photon flux (molecules/photon/sec)
        # F = (sig_ret/hc)*I*lamda/w_loss*1e-9;   % Photon flux (molecules/photon/sec)

        # Light sensitive rates for C1->01 and C2->O2 transition
        k1 = eps1 * F * S0
        k2 = eps2 * F * S0

        # Light sensitive O1->02 transitions
        e12d = 0.011
        e12c1 = 0.005
        e12c2 = 0.024
        e12 = e12d + e12c1*np.log(1+I/e12c2)
        # Light sensitive O2->O1 transitions
        e21d = 0.008
        e21c1 = 0.004
        e21c2 = 0.024
        e21 = e21d + e21c1*np.log(1+I/e21c2)

        mat = np.array([[Gd1 + e12, e12, Gd1, 0],
                        [e21, -(Gd2 + e21), 0, Gd2],
                        [k1, 0, k1, 0],
                        [0, k2, Gd2, (k2 + Gr)]])

        import scipy.linalg
        eigen = scipy.linalg.eig(mat * .01 - np.eye(4), left=True)
        eigen = eigen[0]
        #print eigen
        stationary = eigen / np.sum(np.array(list(eigen)) ** 2)

        ans['O1'] = stationary[0]
        ans['O2'] = stationary[1]
        ans['C1'] = stationary[2]
        ans['C2'] = stationary[3]

        return ans
        
    def IV_plot(self, start = 0, stop = 2000):
        dt = np.dtype([(self.compartment.name, [('V', np.float64), ('[Ca]', np.float64), (self.name, self.latent_dtype)])])

        import matplotlib.pyplot as plt
        from matplotlib import cm
        # from mpl_toolkits.mplot3d import Axes3D

        Irr = np.linspace(0, 700, 100)
        V   = np.linspace(-500, 1000, 100)

        X, Y = np.meshgrid(Irr, V)
        Z = np.zeros(X.shape)
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                latent = np.ndarray(
                    buffer = np.hstack((
                        np.array([[Y[row, col], 0]]),
                        np.array([self.stationary(X[row, col], Y[row, col]).tolist()])
                    )),
                    dtype = dt,
                    shape = dt.shape
                )
                Z[row, col] = self.evaluate_state(np.array([latent]), ())[0][0]

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

        #ax.contour(X, Y, Z, zdir='x', cmap=cm.coolwarm)
        #ax.contour(X, Y, Z, zdir='y', cmap=cm.coolwarm)
        #ax.contour(X, Y, Z, zdir='z', cmap=cm.coolwarm)

        ax.set_xlabel('Irr')
        ax.set_ylabel('V')
        plt.title(self.name)
        plt.show()
