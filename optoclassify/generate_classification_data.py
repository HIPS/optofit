"""
Generate a synthetic dataset of Hodgkin-Huxley style neurons.
We'll use this to test our classification algorithms.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

import cPickle, gzip

from hips.plotting.sausage import sausage_plot

from hh_dynamics import forward_euler, implicit_euler

def make_mixture_distribution():
    # The two classes of neurons are distinguished
    # by differences in the conductance of their
    # leak, sodium, and potassium channels.

    # Since the conductances are nonnegative,
    # we model the channel conductance distribution as
    # a mixture of gamma distributions
    a         = 50.0
    a_healthy = np.array([a,     a,      a])
    b_healthy = np.array([a/0.2, a/120., a/36.0])
    a_disease = np.array([a,     a,      a])
    b_disease = np.array([a/0.2, a/90.,  a/36.0])

    p_healthy = gamma(a=a_healthy, scale=1./b_healthy)
    p_disease = gamma(a=a_disease, scale=1./b_disease)

    return p_healthy, p_disease

def plot_mixture_distribution():
    # Plot the two distributions along the g_Na and g_K axes
    p_healthy, p_disease = make_mixture_distribution()

    # Make a meshgrid
    Gleak, GNa, GK = np.meshgrid(np.linspace(0.1,1.0,10),     # leak conductances
                                 np.linspace(100,150,10),     # Na conductances
                                 np.linspace(20,40,10)        # K conductances
                                 )

    # Compute the healthy cell pdf
    pdf_healthy = p_healthy.pdf(
        np.concatenate((Gleak[:,:,:,None], GNa[:,:,:,None], GK[:,:,:,None]), axis=-1))

    # Compute product of marginal densities
    pdf_healthy = pdf_healthy.prod(axis=-1)

    # Compute the diseased cell pdf
    pdf_disease = p_disease.pdf(
        np.concatenate((Gleak[:,:,:,None], GNa[:,:,:,None], GK[:,:,:,None]), axis=-1))

    # Compute product of marginal densities
    pdf_disease = pdf_disease.prod(axis=-1)

    # Plot
    plt.figure()
    plt.subplot(121)
    plt.imshow(pdf_healthy[:,0,:], extent=(20,40,100,150), origin="lower", interpolation="bilinear", cmap="Greys")
    # plt.contour(GNa[:,0,:], GK[:,0,:], pdf_healthy[:,0,:], 10, interpolation="cubic")
    plt.subplot(122)
    plt.imshow(pdf_disease[:,0,:], extent=(20,40,100,150), origin="lower", interpolation="bilinear", cmap="Greys")
    # plt.contour(GNa[:,0,:], GK[:,0,:], pdf_disease[:,0,:], 10, interpolation="cubic")
    plt.show()

def make_input(t, duty=50., wait=50., amp=75.):
    # Define a pulsed impulse
    inpt  = np.zeros_like(t)
    scale = 1
    offset = 0
    while offset < t[-1]:
        offset += wait
        t_on = offset
        t_off = offset + duty
        inpt  += scale*amp * (t > t_on)  * (t < t_off)

        offset += duty
        scale += 1

    f_inpt = lambda tf: np.interp(tf, t, inpt)
    return f_inpt

def generate_data():

    # Set timing and number of neurons
    N  = 100
    dt = 0.01
    T  = int(1000 / dt)
    t  = np.arange(T) * dt

    # Define a pulsed impulse
    inpt  = make_input(t)

    # Make the channel conductance distributions
    p_healthy, p_disease = make_mixture_distribution()

    # Set the initial conditions shared by both cells
    z0 = np.zeros(4)
    z0[0] = -77.
    z0[1:] = 0.01

    # Simulate N traces of healthy cells
    g_healthy = np.zeros((3,N))
    v_healthy = np.zeros((T,N))
    for n in xrange(N):
        if n % 10 == 0:
            print "Simulating healthy neuron ", n
        # Sample conductances
        g = p_healthy.rvs()
        g_healthy[:,n] = g
        # Simulate a trace
        # z = forward_euler(z0, g, inpt, T, dt)
        z = implicit_euler(z0, g, inpt, t)
        v_healthy[:,n] = z[:,0]

    # Simulate N traces of disease cells
    g_disease = np.zeros((3,N))
    v_disease = np.zeros((T,N))
    for n in xrange(N):
        if n % 10 == 0:
            print "Simulating disease neuron ", n
        # Sample conductances
        g = p_disease.rvs()
        g_disease[:,n] = g
        # Simulate a trace
        # z = forward_euler(z0, g, inpt, T, dt)
        z = implicit_euler(z0, g, inpt, t)
        v_disease[:,n] = z[:,0]

    v_healthy, g_healthy = check_data(v_healthy, g_healthy)
    v_disease, g_disease = check_data(v_disease, g_disease)

    # Compute statistics
    mu_healthy  = v_healthy.mean(axis=1)
    std_healthy = v_healthy.std(axis=1)
    mu_disease  = v_disease.mean(axis=1)
    std_disease = v_disease.std(axis=1)

    plt.figure()
    # Plot disease
    plt.plot(t, mu_disease, color='r', lw=2)
    sausage_plot(t, mu_disease, std_disease, facecolor='r', alpha=0.25)
    plt.plot(t, v_disease, color='r', lw=0.25)

    # Plot healthy
    plt.plot(t, mu_healthy, color='b', lw=2)
    sausage_plot(t, mu_healthy, std_healthy, facecolor='b', alpha=0.25)
    plt.plot(t, v_healthy, color='b', lw=0.25)

    # Scatter plot the conductances
    plt.figure()
    plt.plot(g_healthy[1,:], g_healthy[2,:], 'bo', markerfacecolor='b', markeredgecolor="none", markersize=4)
    plt.plot(g_disease[1,:], g_disease[2,:], 'ro', markerfacecolor='r', markeredgecolor="none", markersize=4)

    plt.show()

    # Save the data
    data = g_healthy, v_healthy, g_disease, v_disease
    with gzip.open("hh_data.pkl.gz", "w") as f:
        cPickle.dump(data, f, protocol=-1)

def check_data(v,g):
    """
    Plot each trace and accept or reject by eye
    :param v:
    :param g:
    :return:
    """
    N = v.shape[1]
    assert g.shape[1] == N

    to_keep = np.ones(N, dtype=np.bool)
    plt.figure()
    plt.ion()
    for n in xrange(N):
        plt.clf()
        plt.plot(v[:,n])
        plt.show()
        accept = raw_input("Accept %d? " % n)
        if accept.strip().lower() != "y":
            print "Rejecting ", n
            to_keep[n] = 0

    plt.ioff()

    print "Accepted: "
    print to_keep

    vp = v[:,to_keep]
    gp = g[:,to_keep]

    return vp, gp


# plot_mixture_distribution()
generate_data()