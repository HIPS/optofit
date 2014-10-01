"""
Placeholder for future container that implements connectivity among populations
of neurons.
"""
import numpy as np
from pybiophys.models.component import Component

class Population(Component):
    """
    Placeholder for future container that implements connectivity among populations
    of neurons.
    """
    def __init__(self, name, model):
        """
        TODO: PyCharm started doing automatic documentation placeholders... cool.

        :return:
        """
        super(Population, self).__init__()
        self._name = name
        self._parent = None

        self.model = model
        self.neurons = self._children

        # TODO: Represent connectivity among the neurons in the population
        self.connectivity = None

        self._latent_dtype = []
        self._state_dtype = []
        self._input_dtype = []
        self._latent_lb = []
        self._latent_ub = []

    def add_neuron(self, neuron):
        """
        Add a compartment to the neuron with the specified electical
        coupling to other compartments

        :param neuron: The neuronto be added
        :return:
        """
        self.neurons.append(neuron)

        # Make a description of the latent state variables
        # Each set of latent variables is accessed by the neuron name
        if neuron.latent_dtype is not None:
            self._latent_dtype.append((neuron.name, neuron.latent_dtype))

        # Make a description of the state variables
        # Each set of state variables is accessed by the compartment name
        if neuron.state_dtype is not None:
            self._state_dtype.append((neuron.name, neuron.state_dtype))

        # Make a description of the input variables
        # Each set of input variables is accessed by the compartment name
        if neuron.input_dtype is not None:
            self._input_dtype.append((neuron.name, neuron.input_dtype))

        # Make array of lower and upper bounds on params
        if neuron.latent_dtype is not None:
            self._latent_lb = np.concatenate((self.latent_lb, neuron.latent_lb))
            self._latent_ub = np.concatenate((self.latent_ub, neuron.latent_ub))

    # TODO: HACK: The dtypes are laid out such that the population is the root, but
    # the path property would make the model the root. Override path property
    # to work around this issue
    @property
    def path(self):
        if self.parent is None:
            return []

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
        for neuron in self.neurons:
            xss = np.concatenate((xss, neuron.steady_state()))
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
        for neuron in self.neurons:
            state[neuron.name] = neuron.evaluate_state(latent, inpt)

        return state

    def kinetics(self, latent, inpt):
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

        # First compute auxiliary state variables like currents under x and y
        state = self.evaluate_state(latent, inpt)

        dxdt = np.zeros(latent.shape, dtype=self.latent_dtype)
        for neuron in self.neurons:
            dxdt[neuron.name] = neuron.kinetics(latent, inpt, state)

        return dxdt

