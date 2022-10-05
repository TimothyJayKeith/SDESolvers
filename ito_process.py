import numpy as np

def ito_approx(a, b, init_val=0, T=1, N=1):
    """
    Approximates the ito process using the Euler approximation

    Parameters
    ----------
    a : drift
    b : diffusion
    init_val : intial condition
    T : length of time interval
    N : number of time steps
    
    Return
    ------
    val_dict: dictionary of values at each time step
    """
    Delta = T/N
    Y = init_val
    val_dict = {0: Y}
    for n in range(N):
        Y += a(Y)*Delta + b(Y)*np.random.normal(0, Delta)
        val_dict[(n+1)*Delta] = Y
    return val_dict

if __name__ == "__main__":
    print(ito_approx(lambda x : 1, lambda x : x, N=5))