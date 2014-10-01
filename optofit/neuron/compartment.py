"""
Base class for a neuron. A neuron consists of a set of compartments, each of which
 has a set of channels. The compartments can be connected by a set of passive resistances.
"""
import numpy as np

from pybiophys.models.component import Component
from pybiophys.models.hyperparameters import hypers

from pybiophys.utils.utils import get_item_at_path

def make_compartment(neuron, model):
    """
    Placeholder for a more complex model that makes different types of compartments.
    """
    if model['compartment_type'].lower() == 'compartment':
        return Compartment(neuron, model)
    elif model['compartment_type'].lower() == 'calcium':
        return CalciumCompartment(neuron, model)
    else:
        raise Exception("Unrecognized compartment type: %s" % model['compartment_type'])

class Compartment(Component):

    def __init__(self, name, neuron, C=None):
        super(Compartment, self).__init__()
        self._name = name
        self.parent = neuron
        self.neuron = self.parent
        self.channels = self.children

        self._latent_dtype = [('V', np.float64)]
        self._state_dtype = [('V', np.float64)]
        self._input_dtype = [('I', np.float64)]
        self._latent_lb = np.array([-np.Inf])
        self._latent_ub = np.array([np.Inf])

        # Set the hyperparameters
        if C is None:
            self.C = hypers['C']

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

    def add_channel(self, channel):
        """
        Add a channel to the compartment's list.

        :param channel:
        :return:
        """
        self.channels.append(channel)

        # Add the channel's properties to that of the compartment
        c = channel
        if c.latent_dtype is not None:
            self._latent_dtype.append((c.name, c.latent_dtype))

        if c.state_dtype is not None:
            self._state_dtype.append((c.name, c.state_dtype))

        if c.input_dtype is not None:
            self._input_dtype.append((c.name, c.input_dtype))

        if c.latent_dtype is not None:
            self._latent_lb = np.concatenate((self.latent_lb, c.latent_lb))
            self._latent_ub = np.concatenate((self.latent_ub, c.latent_ub))


    def steady_state(self):
        """
        Compute steady state of latent variables. This is used
        as the mean of the initial distribution
        """
        # TODO: Grab this from E_leak? Or set E_leak to this?
        dt = np.dtype([('V', np.float64)])
        state = np.ndarray(buffer = np.array([-65.0]), dtype = dt, shape = dt.shape)

        xss = np.array([-65.0])
        for c in self.channels:
            xss = np.concatenate((xss, c.steady_state(state)))

        return xss


    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of this compartment
        """
        state = np.zeros(latent.shape, dtype=self.state_dtype)
        state['V'] = get_item_at_path(latent, self.path)['V']
        for c in self.channels:
            state[c.name] = c.evaluate_state(latent, inpt)

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

        s_comp = get_item_at_path(latent, self.path)
        i_comp = get_item_at_path(inpt, self.path)

        # To compute dV/dt we need the ionic current in this compartment
        I_ionic = 0
        for c in self.channels:
            s_ch = get_item_at_path(state, c.path)
            I_ionic += c.g.value * s_ch['I']

        dxdt['V'] = -1.0/self.C.value * I_ionic

        # TODO Figure out how to handle coupling with other compartments
        # dxdt['V'] += 1.0/self.neuron.C*self.neuron.W(k,:)*(V-Vk)

        # Add in driving current
        dxdt['V'] += 1.0/self.C.value * i_comp['I']

        for c in self.channels:
            if not c.latent_dtype == []:
                dxdt[c.name] = c.kinetics(latent, inpt, state)

        return dxdt

class CalciumCompartment(Compartment):
    def __init__(self, name, neuron, C = None):
        super(CalciumCompartment, self).__init__(name, neuron, C)
        # ranges from 175 to 10
        self.Phi = 100
        # ranges from .001 to .05
        self.Beta = .05

        self._latent_dtype = [('V', np.float64), ('[Ca]', np.float64)]
        self._state_dtype  = [('V', np.float64), ('[Ca]', np.float64)]
        self._state_vars  = [('V', np.float64), ('[Ca]', np.float64)]
        self._input_dtype  = [('I', np.float64) , ('Irr', np.float64)]

        self._latent_lb = np.array([-np.Inf, 0])
        self._latent_ub = np.array([np.Inf, np.Inf])

    def steady_state(self):
        """
        Compute steady state of latent variables. This is used
        as the mean of the initial distribution
        """
        # TODO: Grab this from E_leak? Or set E_leak to this?
        # Add in calcium steady states
        dt = np.dtype(self._state_vars)

        state = np.array([-65, 100 * 10 ** (-9)]).view(dt)
        #state = np.ndarray(buffer = np.array([-65.0, 100 * 10 ** (-9)]), dtype = dt, shape = dt.shape)

        xss = np.array([-65.0, 100 * 10 ** (-9)])
        for c in self.channels:
            xss = np.concatenate((xss, c.steady_state(state)))

        return xss

    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of this compartment
        """
        N = len(latent)
        state = np.zeros(N, dtype=self.state_dtype)
        my_latent = get_item_at_path(latent, self.path)
        state['V']    = my_latent['V']
        state['[Ca]'] = my_latent['[Ca]']

        for c in self.channels:
            if c.latent_dtype is not None:
                state[c.name] = c.evaluate_state(latent, inpt)

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

        s_comp = get_item_at_path(latent, self.path)
        i_comp = get_item_at_path(inpt, self.path)

        # To compute dV/dt we need the ionic current in this compartment
        
        I_ionic   = 0
        I_calcium = 0
        for c in self.channels:
            s_ch = get_item_at_path(state, c.path)
            I_ionic += c.g.value * s_ch['I']

            if c.moves_calcium:
                I_calcium += c.g.value * s_ch['I']

        dxdt['V'] = -1.0/self.C.value * I_ionic

        # We model [Ca] as per page 83 of Traub 1994
        dxdt['[Ca]'] = -1 * I_calcium * self.Phi - s_comp['[Ca]'] * self.Beta

        # TODO Figure out how to handle coupling with other compartments
        # dxdt['V'] += 1.0/self.neuron.C*self.neuron.W(k,:)*(V-Vk)

        # Add in driving current
        dxdt['V'] += 1.0/self.C.value * i_comp['I']

        for c in self.channels:
            if not (c.latent_dtype is None or c.latent_dtype == []):
                tmp = c.kinetics(latent, inpt, state)
                dxdt[c.name] = tmp

        return dxdt
