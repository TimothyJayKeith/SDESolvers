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


def euler_ito_approx(a, b, init_val=0, T=1, N=1, return_full_sim=False):
    """
    Approximates the ito process using the Euler approximation

    Parameters
    ----------
    a : lambda
        The "drift" of the process, or the deterministic portion
    b : lambda
        The "diffusion" of the process, or the random portion
    init_val : float
        The initial condition of the process
    T : positive double
        The length of the time interval (time is assumed to always
        start at 0)
    N : positive int
        The number of time steps
    return_full_sim : bool
        If set to True, will return a dictionary with the value of
        the process at each time step. Otherwise only returns the
        final value of the process
    
    Returns
    -------
    val_dict: dictionary
        Keys are the time steps, values are the values at the time
        step. Only returns if return_full_sim=True
    Y: float
        Value of the process at the final time step
    """
    Delta = T/N
    Y = init_val
    val_dict = {0: Y}
    for n in range(N):
        Y += a(Y)*Delta + b(Y)*np.random.normal(0, Delta)
        val_dict[(n+1)*Delta] = Y
    if return_full_sim:
        return val_dict
    else:
        return Y


def milstein_ito_approx(a, b, init_val=0, T=1, N=1, return_full_sim=False):
    """
    Approximates the ito process using the Milstein scheme

    Parameters
    ----------
    a : lambda
        The "drift" of the process, or the deterministic portion
    b : lambda
        The "diffusion" of the process, or the random portion
    init_val : float
        The initial condition of the process
    T : positive double
        The length of the time interval (time is assumed to always
        start at 0)
    N : positive int
        The number of time steps
    return_full_sim : bool
        If set to True, will return a dictionary with the value of
        the process at each time step. Otherwise only returns the
        final value of the process
    
    Returns
    -------
    val_dict: dictionary
        Keys are the time steps, values are the values at the time
        step. Only returns if return_full_sim=True
    Y: float
        Value of the process at the final time step
    """
    Delta = T/N
    Y = init_val
    val_dict = {0: Y}
    for n in range(N):
        Delta_W = np.random.normal(0, Delta)
        Y += a(Y)*Delta + b(Y)*Delta_W + (1/2)*b(Y)*nderiv(b, Y)*(Delta_W**2 - Delta)
        val_dict[(n+1)*Delta] = Y
    if return_full_sim:
        return val_dict
    else:
        return Y


def strong_taylor_ito_approx(a, b, init_val=0, T=1, N=1, return_full_sim=False):
    """
    Approximates the ito process using the Milstein scheme

    Parameters
    ----------
    a : lambda
        The "drift" of the process, or the deterministic portion
    b : lambda
        The "diffusion" of the process, or the random portion
    init_val : float
        The initial condition of the process
    T : positive double
        The length of the time interval (time is assumed to always
        start at 0)
    N : positive int
        The number of time steps
    return_full_sim : bool
        If set to True, will return a dictionary with the value of
        the process at each time step. Otherwise only returns the
        final value of the process
    
    Returns
    -------
    val_dict: dictionary
        Keys are the time steps, values are the values at the time
        step. Only returns if return_full_sim=True
    Y: float
        Value of the process at the final time step
    """
    Delta = T/N
    Y = init_val
    val_dict = {0: Y}
    for n in range(N):
        # Defining random variables
        zeta_1 = np.random.normal(0, 1)
        zeta_2 = np.random.normal(0, 1)
        Delta_W = zeta_1*Delta**(0.5)
        Delta_Z = (1/2)*(zeta_1 + 1/np.sqrt(3)*zeta_2)*Delta**(1.5)

        # Defining terms in the iteration
        term_1 = a(Y)*Delta + b(Y)*Delta_W + (1/2)*b(Y)*nderiv(b, Y)*(Delta_W**2 - Delta)
        term_2 = b(Y)*nderiv(a, Y)*Delta_Z + (1/2)*(a(Y)*nderiv(a, Y) + (1/2)*b(Y)**2*nderiv2(a, Y))*Delta**2
        term_3 = (a(Y)*nderiv(b, Y) + (1/2)*b(Y)**2*nderiv2(b, Y))*(Delta_W*Delta - Delta_Z)
        term_4 = (1/2)*b(Y)*(b(Y)*nderiv2(b, Y) + nderiv(b, Y)**2)*((1/3)*Delta_W**2 - Delta)*Delta_W

        # Bringing it all together
        Y += term_1 + term_2 + term_3 + term_4
        val_dict[(n+1)*Delta] = Y

    if return_full_sim:
        return val_dict
    else:
        return Y
