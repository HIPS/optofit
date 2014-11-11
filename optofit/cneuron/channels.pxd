from cython.parallel cimport prange

from component cimport Component
from compartment cimport Compartment

cdef class Channel(Component):
    """
    Abstract base class for an ion channel.
    """
    cdef public double g
    cdef public double E

    cdef public Compartment parent_compartment

    cpdef double current(self, double[:,:,::1] x, double V, int t, int n)


cdef class LeakChannel(Channel):
    pass

cdef class NaChannel(Channel):
    pass

cdef class KdrChannel(Channel):
    pass

cdef class GpChannel(Channel)
    pass