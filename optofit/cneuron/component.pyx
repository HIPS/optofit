# distutils: extra_compile_args = -O3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True

import abc

cdef class Component(object):
    """
    Base class for the components of a model
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name=None):
        self.name = name
        self.children = []
        self.parent = None

    # @property
    # def name(self):
    #     return self._name
    #
    # @name.setter
    # def name(self, value):
    #     self._name = value

    # @property
    # def children(self):
    #     return self._children
    #
    # @property
    # def parent(self):
    #     return self._parent

    def add_child(self, child):
        assert isinstance(child, Component), "Child must also be a component!"
        self.children.append(child)

        # Set the child's parent pointer
        child.parent = self

    def initialize_offsets(self, x_base=0, i_base=0):
        """
        Recursively set the offsets into the state and input buffers
        """
        # Initialize the offsets
        x_offset = x_base
        i_offset = i_base

        # Add this component's variables to the offset
        self.x_offset = x_offset
        x_offset += self.n_x
        self.i_offset = i_offset
        i_offset += self.n_i

        # Iterate over children
        for child in self.children:
            ch_n_x, ch_n_i = child.initialize_offsets(x_offset, i_offset)
            x_offset += ch_n_x
            i_offset += ch_n_i

        # Return the number of state variables and inputs used by this component and its children
        return x_offset - x_base, i_offset - i_base

    def steady_state(self, x0):
        """
        Set the steady state

        x0:    a buffer into which the steady state should be placed
        """
        pass

    cpdef kinetics(self, double[:,:,::1] dxdt, double[:,:,::1] x, double[:,::1] inpt, int[::1] t):
        """
        Compute the state kinetics, d{latent}/dt, according to the Hodgkin-Huxley eqns,
        given current state x and external or given variables y.

        latent:  latent state variables of the neuron, e.g. voltage per compartment,
                 channel activation variables, etc.

        inpt:    observations, e.g. supplied irradiance, calcium concentration, injected
                 current, etc.

        state:   evaluated state of the neuron including channel currents, etc.
        """
        pass