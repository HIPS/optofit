"""
A neuron is comprised of a set of connected compartments.
Each compartment contains a set of channels. The purpose
of this class is to provide a coherent and modular framework
for evaluating the Hodgkin-Huxley dynamics of a fairly
simple neuron model.

We are certainly reinventing the wheel here, but it might be
easier than linking into, say, NEURON, at least in the short
term.

The neuron is characterized by a set of latent state variables including:
 - channel activations
 - compartment voltages
 - etc.

It exposes a set of state variables that we might care about, including:
 - channel currents
 - compartment voltages

In order to compute these values, it takes a set of inputs, including:
 - Input currents to each compartment
 - Irradiance shone upon each compartment

The general strategy for storing the latent variables, state variables, and
inputs is to collapse them in a hierarchical numpy structured array. Each
module (neuron, compartment, channel, ...) exposes a description of its
variable types. These are concatenate to compose an overall descriptor for
a top level structured array.

Scott Linderman
5.4.2014
"""
import numpy as np
from optofit.models.component import Component

class Neuron(Component):

    def __init__(self, name, population=None):
        super(Neuron, self).__init__()
        self.name = name
        self.parent = population
        self.population = self.parent
        self.compartments = self.children

        self._latent_dtype = []
        self._state_dtype = []
        self._input_dtype = []
        self._latent_lb = []
        self._latent_ub = []

    def add_compartment(self, compartment, coupling):
        """
        Add a compartment to the neuron with the specified electical
        coupling to other compartments

        :param compartment: The compartment to be added
        :param coupling: The electrical coupling (i.e. resistance or conductance)
        to other compartments already in the model.
        :return:
        """
        self.compartments.append(compartment)

        # TODO: Implement coupling

        # TODO: I'm not particularly pleased with the numpy struct representation for
        # the neuron parameters, states, inputs, bounds, etc. But for now, I'm sticking
        # with it.
        # Make a description of the latent state variables
        # Each set of latent variables is accessed by the compartment name
        c = compartment
        if c.latent_dtype is not None:
            self._latent_dtype.append((c.name, c.latent_dtype))

        # Make a description of the state variables
        # Each set of state variables is accessed by the compartment name
        if c.state_dtype is not None:
            self._state_dtype.append((c.name, c.state_dtype))

        # Make a description of the input variables
        # Each set of input variables is accessed by the compartment name
        if c.input_dtype is not None:
            self._input_dtype.append((c.name, c.input_dtype))

        # Make array of lower and upper bounds on params
        if c.latent_dtype is not None:
            self._latent_lb = np.concatenate((self.latent_lb, c.latent_lb))
            self._latent_ub = np.concatenate((self.latent_ub, c.latent_ub))

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

    def steady_state(self):
        """
        Compute steady state of latent variables. This is used
        as the mean of the initial distribution
        """
        xss = np.array([])
        for c in self.compartments:
            xss = np.concatenate((xss, c.steady_state()))
        return xss

    def evaluate_state(self, latent, inpt):
        """
        Evaluate the state of the neuron given the latent variables x and the
        observed/control signals y.

        latent:  N array of latent state variables of type latent_dtype
                 N is the number of "particles" or latent states to evaluate

        input:  either 1- or N-array of inputs
                if 1, use the same input for each latent state,
                if N, match each input with each latent state
        """
        N = len(latent)
        N_in = len(inpt)
        assert N_in == 1 or N_in == N

        state = np.zeros(latent.shape, dtype=self.state_dtype)
        for c in self.compartments:
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

        returns:
        dxdt:   Rate of change of the latent state variables.
        """
        N = len(latent)
        N_in = len(inpt)
        assert N_in == 1 or N_in == N

        dxdt = np.zeros(latent.shape, dtype=self.latent_dtype)
        for c in self.compartments:
            dxdt[c.name] = c.kinetics(latent, inpt, state)

        return dxdt
