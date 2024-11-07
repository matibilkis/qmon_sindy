import numpy as np
from numba import jit

def dot(a, b):
    return np.einsum('ijk,ikl->ijl', a, b)

## Robler - see sdeint ##
def Aterm(N, h, m, k, dW):
    """kth term in the sum of Wiktorsson2001 equation (2.2)"""
    sqrt2h = np.sqrt(2.0/h)
    Xk = np.random.normal(0.0, 1.0, (N, m, 1))
    Yk = np.random.normal(0.0, 1.0, (N, m, 1))
    term1 = dot(Xk, (Yk + sqrt2h*dW).transpose((0, 2, 1)))
    term2 = dot(Yk + sqrt2h*dW, Xk.transpose((0, 2, 1)))
    return (term1 - term2)/k

## Robler - see sdeint ##
def Ikpw(dW, h, n=5):
    """matrix I approximating repeated Ito integrals for each of N time
    intervals, based on the method of Kloeden, Platen and Wright (1992).
    Args:
      dW (array of shape (N, m)): giving m independent Weiner increments for
        each time step N. (You can make this array using sdeint.deltaW())
      h (float): the time step size
      n (int, optional): how many terms to take in the series expansion
    Returns:
      (A, I) where
        A: array of shape (N, m, m) giving the Levy areas that were used.
        I: array of shape (N, m, m) giving an m x m matrix of repeated Ito
        integral values for each of the N time intervals.
    """
    N = dW.shape[0]
    m = dW.shape[1]
    if dW.ndim < 3:
        dW = dW.reshape((N, -1, 1)) # change to array of shape (N, m, 1)
    if dW.shape[2] != 1 or dW.ndim > 3:
        raise(ValueError)
    A = Aterm(N, h, m, 1, dW)
    for k in range(2, n+1):
        A += Aterm(N, h, m, k, dW)
    A = (h/(2.0*np.pi))*A
    I = 0.5*(dot(dW, dW.transpose((0,2,1))) - np.diag(h*np.ones(m))) + A
    dW = dW.reshape((N, -1)) # change back to shape (N, m)
    return (A, I)

@jit(nopython=True)
def Robler_step(t, Yn, Ik, Iij, dt, f,G, d, m,fext ):
    """
    https://pypi.org/project/sdeint/
    https://dl.acm.org/doi/abs/10.1007/s10543-005-0039-7
    """
    fnh = f(Yn, t,dt)*dt + fext*dt# shape (d,)
    xicov = Gn = G()
    sum1 = np.dot(Gn, Iij)/np.sqrt(dt) # shape (d, m)

    H20 = Yn + fnh # shape (d,)
    H20b = np.reshape(H20, (d, 1))
    H2 = H20b + sum1 # shape (d, m)

    H30 = Yn
    H3 = H20b - sum1
    fn1h = f(H20, t, dt)*dt + fext*dt
    Yn1 = Yn + 0.5*(fnh + fn1h) + np.dot(xicov, Ik)
    return Yn1


@jit(nopython=True)
def Euler(f,g,x,dWs,i,t,dt):
    return f(x[:,i],t)*dt + g(x[:,i],t).dot(dWs[:,i])

@jit(nopython=True)
def RK4(f,g,x,dWs,i,t,dt):
    ###https://people.math.sc.edu/Burkardt/cpp_src/stochastic_rk/stochastic_rk.cpp
    #Runge-Kutta Algorithm for the Numerical Integration
    #of Stochastic Differential Equations
    ##
    a21 =   0.66667754298442
    a31 =   0.63493935027993
    a32 =   0.00342761715422#D+00
    a41 = - 2.32428921184321#D+00
    a42 =   2.69723745129487#D+00
    a43 =   0.29093673271592#D+00
    a51 =   0.25001351164789#D+00
    a52 =   0.67428574806272#D+00
    a53 = - 0.00831795169360#D+00
    a54 =   0.08401868181222#D+00

    q1 = 3.99956364361748#D+00
    q2 = 1.64524970733585#D+00
    q3 = 1.59330355118722#D+00
    q4 = 0.26330006501868#D+00

    t1 = t
    x1 = x[:,i]
    dw=dWs[:,i]
    k1 = dt*f(x1,t) + np.sqrt(q1)*g(x1,t).dot(dw)

    t2 = t1 + (a21 * dt)
    x2 = x1 + (a21 * k1)
    k2 = dt * f(x2,t2) + np.sqrt(q2)*g(x2,t2).dot(dw)

    t3 = t1 + (a31 * dt)  + (a32 * dt)
    x3 = x1 + (a31 * k1) + (a32 * k2)
    k3 = dt * f(x3,t3 ) + np.sqrt(q3)*g(x3, t3).dot(dw)

    t4 = t1 + (a41 * dt)  + (a42 * dt)  + (a43 * dt)
    x4 = x1 + (a41 * k1) + (a42 * k2) + (a43 * k3)
    k4 = dt * f( x4,t4) + np.sqrt(q4)*g(x4,t4).dot(dw)

    return (a51 * k1) + (a52 * k2) + (a53 * k3) + (a54 * k4)
