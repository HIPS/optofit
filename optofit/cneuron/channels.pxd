from cython.parallel cimport prange

import numpy as np
cimport numpy as np

from component cimport Component
from compartment cimport Compartment

from libc.math cimport exp, log

#import GPy

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

cdef class GPChannel(Channel):
    cdef public int grid
    cdef public double z_min
    cdef public double z_max
    cdef public double V_min
    cdef public double V_max

    cdef public kernel_z
    cdef public kernel_V
    cdef public kernel

    cpdef public Z
    cpdef public h
    cpdef public gp

cdef inline double sigma(double z): 1./(1+exp(-z))
cdef inline double sigma_inv(double u): log(u/(1.0-u))