import numpy as np

def nderiv(f, x, h=2.0**(-7.0)):
    """
    Calculates a numerical derivative

    Parameters
    ----------
    f: lambda
        Function you wish to differentiate
    x: float
        Point at which you wish to differentiate the function
    h: float
        The precision with which the derivative is computed

    Returns
    -------
    Approximate derivative of function f at point x
    """
    return (f(x + h) - f(x - h))/(2*h)


def nderiv2(f, x, h=2.0**(-7.0)):
    """
    Calculates a numerical second derivative

    Parameters
    ----------
    f: lambda
        Function you wish to differentiate
    x: float
        Point at which you wish to differentiate the function
    h: float
        The precision with which the derivative is computed

    Returns
    -------
    Approximate second derivative of function f at point x
    """
    return (f(x + h) - 2*f(x) + f(x - h))/h**2


def sampleW(zeta_1, var):
    """
    Samples randomly from W distribution with random variance

    Parameters
    ----------
    zeta_1: float
        point from a random distribution
    var: float
        variance of distribution
    
    Returns
    -------
        Point randomly sampled from W distribution
    """
    return zeta_1*np.sqrt(var)


def sampleZ(zeta_1, zeta_2, var):
    """
    Samples randomly from W distribution with random variance

    Parameters
    ----------
    zeta_1, zeta_2: float
        points from random distributions
    var: float
        variance of distribution
    
    Returns
    -------
        Point randomly sampled from W distribution
    """
    return (1/2)*(zeta_1 + 1/np.sqrt(3)*zeta_2)*var**(1.5)


def euler_iteration(a, b, Y, Delta, zeta=None):
    """
    Performs a single method of the ito_approximation with the Euler method

    Parameters
    ----------
    a: lambda
        The "drift" or deterministic portion of the equation
    b: lambda
        The "diffusion" or random portion of the equation
    Y: float
        The previous value in the iteration
    Delta: float
        Size of the time step
    zeta: float
        Point from a random distribution. If None, the function will randomly sample a point itself

    Returns
    -------
    Next point in the iteration
    """
    if zeta is None:
        zeta = np.random.normal(0, 1)
    return a(Y)*Delta + b(Y)*sampleW(zeta, Delta)


def milstein_iteration(a, b, Y, Delta, zeta=None):
    """
    Performs a single method of the ito_approximation with the Milstein method

    Parameters
    ----------
    a: lambda
        The "drift" or deterministic portion of the equation
    b: lambda
        The "diffusion" or random portion of the equation
    Y: float
        The previous value in the iteration
    Delta: float
        Size of the time step
    zeta: float
        Point from a random distribution. If None, the function will randomly sample a point itself

    Returns
    -------
    Next point in the iteration
    """
    if zeta is None:
        zeta = np.random.normal(0, 1)
    return euler_iteration(a, b, Y, Delta, zeta) + (1/2)*b(Y)*nderiv(b, Y)*(sampleW(zeta, Delta) - Delta)


def strong_taylor_iteration(a, b, Y, Delta, zeta_1=None, zeta_2=None):
    """
    Performs a single method of the ito_approximation with the Strong Taylor method

    Parameters
    ----------
    a: lambda
        The "drift" or deterministic portion of the equation
    b: lambda
        The "diffusion" or random portion of the equation
    Y: float
        The previous value in the iteration
    Delta: float
        Size of the time step
    zeta: float
        Point from a random distribution. If None, the function will randomly sample a point itself

    Returns
    -------
    Next point in the iteration
    """
    if zeta_1 is None:
        zeta_1 = np.random.normal(0, 1)
    if zeta_2 is None:
        zeta_2 = np.random.normal(0, 1)
    Delta_W = sampleW(zeta_1, Delta)
    Delta_Z = sampleZ(zeta_1, zeta_2, Delta)

    term_list = [milstein_iteration(a, b, Y, Delta, zeta=zeta_1)]
    term_list.append(b(Y)*nderiv(a, Y)*Delta_Z + (1/2)*(a(Y)*nderiv(a, Y) + (1/2)*b(Y)**2*nderiv2(a, Y))*Delta**2)
    term_list.append((a(Y)*nderiv(b, Y) + (1/2)*b(Y)**2*nderiv2(b, Y))*(Delta_W*Delta - Delta_Z))
    term_list.append((1/2)*b(Y)*(b(Y)*nderiv2(b, Y) + nderiv(b, Y)**2)*((1/3)*Delta_W**2 - Delta)*Delta_W)
    return np.sum(term_list)


def ito_approx(a, b, init_val=0.0, T=1.0, N=10, method="euler", return_full_sim=False):
    """
    Simulates the Ito process using any one of a variety of methods

    Parameters
    ----------
    a: lambda
        The "drift" or deterministic portion of the equation
    b: lambda
        The "diffusion" or random portion of the equation
    init_val: float
        The initial value of the ito process
    T: float
        Size of time interval (time is always assumed to start at 0)
    N: positive int
        Number of time steps
    method: string
        Name of desired approximation method. Defaults to Euler method,
        but Milstein and Strong Taylor methods also available.
    return_full_sim: bool
        If set to true, returns a list with the value of the simulation at
        each time step. Otherwise, only returns value at final time step.

    Returns
    -------
    Either value of process at final time step or a list of values at
    each time step, depending on value of return_full_sim.
    """
    Delta = T/N
    Y = [init_val]

    if method in ["euler"]:
        iteration = euler_iteration
    elif method in ["milstein"]:
        iteration = milstein_iteration
    elif method in ["strong taylor", "strong"]:
        iteration = strong_taylor_iteration
    else:
        raise Exception(f"{method} is not a known iteration method")
    
    for n in range(N):
        Y.append(Y[n] + iteration(a, b, Y[n], Delta))
    if return_full_sim:
        return Y
    else:
        return Y[N]
