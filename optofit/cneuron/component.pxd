
cdef class Component(object):
    """
    Base class for the components of a model
    """
    # Define cython class variables
    cdef public int x_offset       # Offset into the latent state buffer
    cdef public int n_x            # Number of latent state values
    cdef public int i_offset       # Offset into the input buffer
    cdef public int n_i            # Number of inputs to this component

    cdef public object name
    cdef public list children
    cdef public Component parent

    # Component cython functions
    cpdef kinetics(self, double[:,:,::1] dxdt, double[:,:,::1] x, double[:,::1] inpt, int[::1] t)