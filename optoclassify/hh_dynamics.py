"""
Simple simulation for the standard Hodgkin-Huxley model of a
squid giant axon. It consists of two channels, a sodium and a
delayed rectification potassium channel.
"""
import numpy as np

# Define a few constants
E_leak = -60.       # Reversal potentials
E_na   = 50.
E_k    = -77.

g_leak = 0.3        # Default Conductances
g_na   = 120
g_k    = 36


#############################################################
# Dynamics functions
#############################################################
def voltage_dynamics(V, m, h, n, t, g, inpt):

    # Compute the voltage drop through each channel
    V_leak = V - E_leak
    V_na = m**3 * h * (V-E_na)
    V_k = n**4 * (V-E_k)

    I_ionic = g[0] * V_leak
    I_ionic += g[1] * V_na
    I_ionic += g[2] * V_k

    # dVdt[t,n] = -1.0/self.C * I_ionic
    dVdt = -1.0 * I_ionic

    # Add in driving current
    dVdt += inpt[t]

    return dVdt

def sodium_dynamics(V, m, h, t):

    # Use resting potential of zero
    V_ref = V + 60

    # Compute the alpha and beta as a function of V
    am1 = 0.32*(13.1-V_ref)/(np.exp((13.1-V_ref)/4)-1)
    ah1 = 0.128 * np.exp((17.0-V_ref)/18.0)

    bm1 = 0.28*(V_ref-40.1)/(np.exp((V_ref-40.1)/5.0)-1.0)
    bh1 = 4.0/(1.0 + np.exp((40.-V_ref)/5.0))

    dmdt = am1*(1.-m) - bm1*m
    dhdt = ah1*(1.-h) - bh1*h

    return dmdt, dhdt

def potassium_dynamics(V, n, t):
    # Use resting potential of zero
    V_ref = V+60

    # Compute the alpha and beta as a function of V
    an1 = 0.01*(V_ref+55.) /(1-np.exp(-(V_ref+55.)/10.))
    bn1 = 0.125 * np.exp(-(V_ref+65.)/80.)

    # Compute the channel state updates
    dndt = an1 * (1.0-n) - bn1*n

    return dndt

#############################################################
# Simulation
#############################################################
def forward_euler(z0, g, inpt, T, dt=1.0):
    z = np.zeros((T,4))

    z[0,:] = z0
    for t in xrange(1,T):
        # Extract state at time t-1
        Vtm1 = z[t-1,0]
        mtm1 = z[t-1,1]
        htm1 = z[t-1,2]
        ntm1 = z[t-1,3]

        # Compute dynamics
        dvdt = voltage_dynamics(Vtm1, mtm1, htm1, ntm1, t, g, inpt)
        dmdt, dhdt = sodium_dynamics(Vtm1, mtm1, htm1, t)
        dndt = potassium_dynamics(Vtm1, ntm1, t)

        # Update state
        z[t,0] = Vtm1 + dt*dvdt
        z[t,1] = mtm1 + dt*dmdt
        z[t,2] = htm1 + dt*dhdt
        z[t,3] = ntm1 + dt*dndt

    return z
