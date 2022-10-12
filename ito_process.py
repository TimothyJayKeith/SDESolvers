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
    return (f(x + h) - 2*f(x) + f(x - h))/(2*h)


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
    if zeta is None:
        zeta = np.random.normal(0, 1)
    return a(Y)*Delta + b(Y)*sampleW(zeta, Delta)


def milstein_iteration(a, b, Y, Delta, zeta=None):
    if zeta is None:
        zeta = np.random.normal(0, 1)
    return euler_iteration(a, b, Y, Delta, zeta) + (1/2)*b(Y)*nderiv(b, Y)*(sampleW(zeta, Delta) - Delta)


def strong_taylor_iteration(a, b, Y, Delta, zeta_1=None, zeta_2=None):
    if zeta_1 is None:
        zeta_1 = np.random.normal(0, 1)
    if zeta_2 is None:
        zeta_2 = np.random.normal(0, 1)
    term_list = [milstein_iteration(a, b, Y, Delta, zeta=zeta_1)]
    term_list.append(b(Y)*nderiv(a, Y)*sampleZ(zeta_1, zeta_2, Delta) + (1/2)*(a(Y)*nderiv(a, Y) + (1/2)*b(Y)**2*nderiv2(a, Y))*Delta**2)
    term_list.append((a(Y)*nderiv(b, Y) + (1/2)*b(Y)**2*nderiv2(b, Y))*(sampleW(Delta, zeta_1)*Delta - sampleZ(Delta, zeta_1, zeta_2)))
    term_list.append((1/2)*b(Y)*(b(Y)*nderiv2(b, Y) + nderiv(b, Y)**2)*((1/3)*sampleW(Delta, zeta_1)**2 - Delta)*sampleW(Delta, zeta_1))
    return np.sum(term_list)


def ito_approx(a, b, init_val=1, T=1.0, N=10, method="euler", return_full_sim=False):
    Delta = T/N
    Y = init_val
    val_dict = {0: Y}

    if method in ["euler"]:
        iteration = euler_iteration
    elif method in ["milstein"]:
        iteration = milstein_iteration
    elif method in ["strong taylor", "strong"]:
        iteration = strong_taylor_iteration
    else:
        raise Exception(f"{method} is not a known iteration method")
    
    for n in range(N):
        Y += iteration(a, b, Y, Delta)
        val_dict[n+1] = Y
    if return_full_sim:
        return val_dict
    else:
        return Y
