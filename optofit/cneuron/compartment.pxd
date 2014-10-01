from cython.parallel cimport prange

from component cimport Component
from channels cimport *

cdef class Compartment(Component):
    """
    Simple compartment model with voltage
    """
    cdef public double C
    cdef public double V0

cdef class SquidCompartment(Compartment):
    """
    Special case compartment wiht leak, na, and kdr channels
    """
    cdef LeakChannel leak
    cdef NaChannel na
    cdef KdrChannel kdr