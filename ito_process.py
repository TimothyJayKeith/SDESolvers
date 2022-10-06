import numpy as np

def ito_approx(a, b, init_val=0, T=1, N=1, return_full_sim=False):
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
