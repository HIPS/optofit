import numpy as np
import matplotlib.pyplot as plt

logit = lambda x: np.log(x / (1-x))
logistic = lambda u: np.exp(u) / (1 + np.exp(u))
dlogit = lambda x: 1./(x*(1-x))

g = lambda x: x**4
ginv = lambda u: u**(1./4)
dg_dx = lambda x: 4*x**3

u_to_x = lambda u: ginv(logistic(u))
x_to_u = lambda x: logit(g(x))

uu = np.linspace(-6,6,1000)
xx = u_to_x(uu)
#g = lambda x: x
#ginv = lambda u: u
#dg_dx = lambda x: 1.0


# Compute dynamics du/dt 
alpha = lambda V: 0.01 * (10.01-V) / (np.exp((10.01-V)/10.) - 1)
beta = lambda V: 0.125 * np.exp(-V/80.)
dx_dt = lambda x,V: alpha(V)*(1-x) - beta(V) * x
du_dt = lambda u,V: dlogit(g(u_to_x(u))) * dg_dx(u_to_x(u)) * dx_dt(u_to_x(u),V)

# Plot the change in u as a function of u and V
V = np.linspace(0,100,100)
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(du_dt(uu[:,None], V[None,:]), 
           extent=[V[0], V[-1], uu[-1], uu[0]], 
           interpolation="none",
           cmap=plt.cm.Reds)
ax1.set_aspect(20)
ax1.set_xlabel('V')
ax1.set_ylabel('u')
ax1.set_title('du_dt(u,V)')

ax2 = fig.add_subplot(1,2,2)
ax2.imshow(dx_dt(xx[:,None], V[None,:]), 
           extent=[V[0], V[-1], xx[-1], xx[0]], 
           interpolation="none",
           cmap=plt.cm.Reds)
ax2.set_aspect(100)
ax2.set_xlabel('V')
ax2.set_ylabel('x')
ax2.set_title('dx_dt(x,V)')
plt.show()

