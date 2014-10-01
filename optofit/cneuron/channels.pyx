# distutils: extra_compile_args = -O3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True

from cython.parallel cimport prange

from component cimport Component
from compartment cimport Compartment

from libc.math cimport exp

cdef class Channel(Component):
    """
    Abstract base class for an ion channel.
    """

    def __init__(self, name=None):
        super(Channel, self).__init__(name=name)

        # All channels (at least so far!) have a conductance and a reversal
        # potential

    cpdef double current(self, double[:,:,::1] x, double V, int t, int n):
        pass

cdef class LeakChannel(Channel):
    """
    Passive leak channel.
    """
    def __init__(self, name=None, hypers=None):
        super(LeakChannel, self).__init__(name=name)

        # Set the hyperparameters
        # TODO: Use distributions to set the hypers?
        self.g = hypers['g_leak']
        self.E = hypers['E_leak']

    cpdef double current(self, double[:,:,::1] x, double V, int t, int n):
        """
        Evaluate the instantaneous current through this channel
        """
        return V - self.E

cdef class NaChannel(Channel):
    """
    Sodium channel.
    """
    def __init__(self, name=None, hypers=None):
        super(NaChannel, self).__init__(name=name)

        # Specify the number of parameters
        self.n_x = 2

        # Set the hyperparameters
        # TODO: Use distributions to set the hypers?
        self.g = hypers['g_na']
        self.E = hypers['E_na']

    def steady_state(self, x0):
        """
        Set the steady state

        x0:    a buffer into which the steady state should be placed
        """
        V = x0[self.parent_compartment.x_offset]
        # Steady state value of the latent vars
        # Compute the alpha and beta as a function of V
        am1 = 0.1*(V+35.)/(1-exp(-(V+35.)/10.))
        ah1 = 0.07*exp(-(V+50.)/20.)
        bm1 = 4.*exp(-(V+65.)/18.)
        bh1 = 1./(exp(-(V+35)/10.)+1)

        x0[self.x_offset+0] = am1/(am1+bm1)
        x0[self.x_offset+1] = ah1/(ah1+bh1)

    cpdef double current(self, double[:,:,::1] x, double V, int t, int n):
        """
        Evaluate the instantaneous current through this channel
        """
        cdef double m = x[t,n,self.x_offset+0]
        cdef double h = x[t,n,self.x_offset+1]
        return m**3 * h * (V - self.E)

    cpdef kinetics(self, double[:,:,::1] dxdt, double[:,:,::1] x, double[:,::1] inpt, int[::1] ts):
        cdef int T = x.shape[0]
        cdef int N = x.shape[1]
        cdef int D = x.shape[2]
        cdef int M = inpt.shape[1]
        cdef int S = ts.shape[0]
        cdef int n, s, t
        cdef double am1, ah1, bm1, bh1

        # Get a pointer to the voltage of the parent compartment
        # TODO: This approach sucks b/c it assumes the voltage
        # is the first compartment state. It should be faster than
        # calling back into the parent to have it extract the voltage
        # for us though.
        # cdef double[:,:] V = x[:,:,self.parent_compartment.x_offset]
        # cdef double[:,:] m = x[:,:,self.x_offset+0]
        # cdef double[:,:] h = x[:,:,self.x_offset+1]
        cdef double V, m, h

        with nogil:
            for s in prange(S):
                t = ts[s]
                for n in prange(N):
                    # # Compute the alpha and beta as a function of V
                    # am1 = 0.1*(V[t,n]+35.)/(1-exp(-(V[t,n]+35.)/10.))
                    # ah1 = 0.07*exp(-(V[t,n]+50.)/20.)
                    #
                    # bm1 = 4.*exp(-(V[t,n]+65.)/18.)
                    # bh1 = 1./(exp(-(V[t,n]+35.)/10.)+1.)
                    #
                    # # Compute the channel state updates
                    # dxdt[t,n,self.x_offset+0] = am1*(1.-m[t,n]) - bm1*m[t,n]
                    # dxdt[t,n,self.x_offset+1] = ah1*(1.-h[t,n]) - bh1*h[t,n]

                    # Compute the alpha and beta as a function of V
                    V = x[t,n,self.parent_compartment.x_offset]
                    m = x[t,n,self.x_offset]
                    h = x[t,n,self.x_offset+1]
                    am1 = 0.1*(V+35.)/(1-exp(-(V+35.)/10.))
                    ah1 = 0.07*exp(-(V+50.)/20.)

                    bm1 = 4.*exp(-(V+65.)/18.)
                    bh1 = 1./(exp(-(V+35.)/10.)+1.)

                    # Compute the channel state updates
                    dxdt[t,n,self.x_offset+0] = am1*(1.-m) - bm1*m
                    dxdt[t,n,self.x_offset+1] = ah1*(1.-h) - bh1*h


cdef class KdrChannel(Channel):
    """
    Potassium (delayed rectification) channel.
    """
    def __init__(self, name=None, hypers=None):
        super(KdrChannel, self).__init__(name=name)

        # Specify the number of parameters
        self.n_x = 1

        # Set the hyperparameters
        # TODO: Use distributions to set the hypers?
        self.g = hypers['g_kdr']
        self.E = hypers['E_kdr']

    def steady_state(self, x0):
        V = x0[self.parent_compartment.x_offset]
        # Steady state value of the latent vars
        an1 = 0.01*(V+55.)/(1-exp(-(V+55.)/10.))
        bn1 = 0.125*exp(-(V+65.)/80.)

        x0[self.x_offset] = an1/(an1+bn1)

    cpdef double current(self, double[:,:,::1] x, double V, int t, int n):
        """
        Evaluate the instantaneous current through this channel
        """
        cdef double nn = x[t,n,self.x_offset]
        return nn**4 * (V - self.E)

    cpdef kinetics(self, double[:,:,::1] dxdt, double[:,:,::1] x, double[:,::1] inpt, int[::1] ts):
        cdef int T = x.shape[0]
        cdef int N = x.shape[1]
        cdef int D = x.shape[2]
        cdef int M = inpt.shape[1]
        cdef int S = ts.shape[0]
        cdef int n, s, t
        cdef double an1, bn1

        # Get a pointer to the voltage of the parent compartment
        # TODO: This approach sucks b/c it assumes the voltage
        # is the first compartment state. It should be faster than
        # calling back into the parent to have it extract the voltage
        # for us though.
        # cdef double[:,:] V = x[:,:,self.parent_compartment.x_offset]
        # cdef double[:,:] nn = x[:,:,self.x_offset+0]

        cdef double V, nn

        with nogil:
            for s in prange(S):
                t = ts[s]
                for n in prange(N):
                    V = x[t,n,self.parent_compartment.x_offset]
                    nn = x[t,n,self.x_offset]

                    # Compute the alpha and beta as a function of V
                    an1 = 0.01*(V+55.) /(1-exp(-(V+55.)/10.))
                    bn1 = 0.125*exp(-(V+65.)/80.)

                    # Compute the channel state updates
                    dxdt[t,n,self.x_offset] = an1*(1.-nn) - bn1*nn


        return dxdt