def char_eq_copy(x, *args):
    """
    This function evaluates the charachteristic equation at a given point.

    Parameters:
        x ([float, float]):
            A list of 2 elements, making up the Re and Im parts of the complex eigenvalue to calculate char_eq.
        args[0]:
            A dictionary for parameters.

    Returns:
        array[float, float]:
            An array of 2 elements, making up the Re and Im parts of the complex value of char_eq at the given x.
    """
    import numpy as np
    from .eig_fun import eig_fun_1_prime, eig_fun_1, eig_fun_2

    par = args[0]

    # Extract real and imaginary parts from the input tuple
    x_real, x_imag = x[0], x[1]

    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])

    # Create complex numbers element-wise from real and imaginary parts
    l = complex(x_real, x_imag)

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    if np.isclose(p_sqrt,0):
        return 1
    
    r = [
        (v+p_sqrt)/(2*D),
        (v-p_sqrt)/(2*D),
    ]

    y = D * eig_fun_1_prime(0, par, l) - v * eig_fun_1(0, par, l) + R * v * eig_fun_2(0, par, l)
    
    # y = D * r[1] * (
    # 1 - np.exp(r[1]-r[0])) - v * (
    # 1 - r[1]/r[0] * np.exp(r[1]-r[0])) + R*v*np.exp(r[1] - t*l) * (
    # 1 - r[1]/r[0])

    return y.real, y.imag