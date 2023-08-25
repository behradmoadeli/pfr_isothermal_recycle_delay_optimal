def char_eq(x, *args):  # To be more simplified
    """
    This function evaluates the charachteristic equation at a given point.

    Parameters:
        x ([float, float]):
            A list of 2 elements, making up the Re and Im parts of the complex eigenvalue to calculate char_eq.

    Returns:
        array[float, float]:
            An array of 2 elements, making up the Re and Im parts of the complex value of char_eq at the given x.
    """
    import numpy as np  # Import numpy inside the function

    par = args[0]
    l = complex(x[0], x[1])

    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    if np.isclose(p_sqrt, 0, atol=1e-8):
        y = (
            np.exp(l*t+v/2/D) * (v**2 + 2*D**2)
            + v * (1-R*np.exp(v/D))
        )
    else:
        y = (
            np.exp(l*t+v/2/D) * np.sinh(np.sqrt(p)/2/D) * (v**2 + 2*D**2)
            + v * np.sqrt(p) * (np.cosh(np.sqrt(p)/2/D)-R*np.exp(v/D))
        )

    return np.array([y.real, y.imag])