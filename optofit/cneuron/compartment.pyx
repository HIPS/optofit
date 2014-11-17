# distutils: extra_compile_args = -O3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True

from cython.parallel cimport prange

from component cimport Component
from component import Component

from channels cimport Channel

import numpy as np
from hips.inference.mh import mh

cdef class Compartment(Component):
    """
    Simple compartment model with voltage
    """
    def __init__(self, name=None, hypers=None):
        super(Compartment, self).__init__(name=name)
        self.C = hypers['C']
        self.V0 = hypers['V0']
        self.n_x = 1
        self.n_i = 1

    def add_child(self, child):
        assert isinstance(child, Channel), "Child must also be a component!"
        super(Compartment, self).add_child(child)

        # Also, make sure the channel has a C pointer to this compartment
        child.parent_compartment = self

    def steady_state(self, x0):
        """
        Set the steady state

        x0:    a buffer into which the steady state should be placed
        """
        # Set the resting potential
        x0[self.x_offset] = self.V0
        for child in self.children:
            child.steady_state(x0)

    cpdef kinetics(self, double[:,:,::1] dxdt, double[:,:,::1] x, double[:,::1] inpt, int[::1] ts):
        cdef int T = x.shape[0]
        cdef int N = x.shape[1]
        cdef int D = x.shape[2]
        cdef int M = inpt.shape[1]
        cdef int S = ts.shape[0]
        cdef int n, s, t
        cdef double V, dVdt

        # Compute the change in voltage for each time and particle
        # cdef double[:,:] dVdt = dxdt[:,:,self.x_offset]

        # We need the GIL because we want to iterate over the children,
        # which, in this generic case, are just a list of Python objects
        # as far as the compiler knows. We can improve performance with
        # special-cased compartments with specific channels
        for s in range(S):
            t = ts[s]
            for n in range(N):
                # To compute dV/dt we need the ionic current in this compartment
                V = x[t,n,self.x_offset]
                I_ionic = 0
                for c in self.children:
                    I_ionic += c.g * c.current(x, V, t, n)
                    # pass

                # dVdt[t,n] = -1.0/self.C * I_ionic
                dVdt = -1.0/self.C * I_ionic

                # TODO Figure out how to handle coupling with other compartments
                # dxdt['V'] += 1.0/self.neuron.C*self.neuron.W(k,:)*(V-Vk)

                # Add in driving current
                # dVdt[t,n] += 1.0/self.C * inpt[t,self.i_offset]
                dVdt += 1.0/self.C * inpt[t,self.i_offset]

                dxdt[t,n,self.x_offset] = dVdt

        # Compute dxdt for each channel
        for c in self.children:
            c.kinetics(dxdt, x, inpt, ts)


    def plot(self, t, z, axs=None, lines=None, currents=True, color='k'):
        """
        Plot the latent state. If we aren't given axes or lines, create a new figure first.
        If we're given the axes but not lines, plot in the existing axes.
        If we're given handles to lines, update the line data.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if currents == False:
            raise NotImplementedError()

        assert z.ndim == 2
        zz = z[:,None,:].copy('C')
        V = z[:,self.x_offset].copy('C')
        T = z.shape[0]
        C = len(self.children)
        # Compute the current
        I = np.zeros((T, C))
        for m,c in enumerate(self.children):
            for i in range(T):
                # I[i,m] = c.g * c.current(zz, V[i], i, 0)
                I[i,m] = c.current(zz, V[i], i, 0)

        M = 1 + C
        if lines is None and axs is None:
            axs = []
            lines = []

            fig = plt.figure()
            ax1 = fig.add_subplot(M,1,1)
            l1 = ax1.plot(t, V, color=color)
            ax1.set_ylabel('V')

            axs.append(ax1)
            lines.append(l1)

            for m,c in enumerate(self.children):
                axm = fig.add_subplot(M,1,m+2)
                lm = axm.plot(t, I[:,m], color=color)

                axm.set_ylabel('I_%s' % c.name)
                axm.set_ylim((-20,20))

                axs.append(axm)
                lines.append(lm)

        elif lines is None and axs is not None:
            lines = []
            assert len(axs) == M, "compartment.plot received incorrect number of axes"
            ax1 = axs[0]
            l1 = ax1.plot(t, z[:,self.x_offset], color=color)

            lines.append(l1)

            for m,c in enumerate(self.children):
                axm = axs[1+m]
                # Compute the current
                V = z[:,self.x_offset]
                lm = axm.plot(t, I[:,m], color=color)

                lines.append(lm)

        elif lines is not None:
            lines[0][0].set_data(t, z[:, self.x_offset])

            for m,c in enumerate(self.children):
                lines[1+m][0].set_data(t, I[:,m])

        return axs, lines


cdef class SquidCompartment(Compartment):
    """
    Special case of the squid compartment studied by
    Hodgkin and Huxley. We use a single compartment
    with leak, sodium, and potassium channels.
    """
    def __init__(self, name=None, hypers=None):
        super(Compartment, self).__init__(name=name)
        self.C = hypers['C']
        self.V0 = hypers['V0']
        self.n_x = 1
        self.n_i = 1

        # Create the channels
        from channels import LeakChannel, NaChannel, KdrChannel
        self.leak = LeakChannel(name='leak', hypers=hypers)
        self.na = NaChannel(name='na', hypers=hypers)
        self.kdr = KdrChannel(name='kdr', hypers=hypers)

        # Add the channels so that the offsets can be set properly
        self.add_child(self.leak)
        self.add_child(self.na)
        self.add_child(self.kdr)

    def steady_state(self, x0):
        """
        Set the steady state

        x0:    a buffer into which the steady state should be placed
        """
        # Set the resting potential
        x0[self.x_offset] = self.V0

        # Rather than iterating over children, in this special case model
        # we can simply call the known channels
        self.leak.steady_state(x0)
        self.na.steady_state(x0)
        self.kdr.steady_state(x0)

    cpdef kinetics(self, double[:,:,::1] dxdt, double[:,:,::1] x, double[:,::1] inpt, int[::1] ts):
        cdef int T = x.shape[0]
        cdef int N = x.shape[1]
        cdef int D = x.shape[2]
        cdef int M = inpt.shape[1]
        cdef int S = ts.shape[0]
        cdef int n, s, t
        cdef double V, dVdt

        # Compute the change in voltage for each time and particle
        # cdef double[:,:] dVdt = dxdt[:,:,self.x_offset]

        # We need the GIL because we want to iterate over the children,
        # which, in this generic case, are just a list of Python objects
        # as far as the compiler knows. We can improve performance with
        # special-cased compartments with specific channels
        for s in range(S):
            t = ts[s]
            for n in range(N):
                # To compute dV/dt we need the ionic current in this compartment
                V = x[t,n,self.x_offset]

                I_ionic = self.leak.g * self.leak.current(x, V, t, n)
                I_ionic += self.na.g * self.na.current(x, V, t, n)
                I_ionic += self.kdr.g * self.kdr.current(x, V, t, n)


                # dVdt[t,n] = -1.0/self.C * I_ionic
                dVdt = -1.0/self.C * I_ionic

                # TODO Figure out how to handle coupling with other compartments
                # dxdt['V'] += 1.0/self.neuron.C*self.neuron.W(k,:)*(V-Vk)

                # Add in driving current
                dVdt += 1.0/self.C * inpt[t,self.i_offset]

                # Set dVdt in the output buffer
                dxdt[t,n,self.x_offset] = dVdt

        # Compute dxdt for each channel
        self.leak.kinetics(dxdt, x, inpt, ts)
        self.na.kinetics(dxdt, x, inpt, ts)
        self.kdr.kinetics(dxdt, x, inpt, ts)