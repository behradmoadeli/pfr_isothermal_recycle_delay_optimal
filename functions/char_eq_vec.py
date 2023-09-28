def char_eq_vec(x, *args):
    """
    This function evaluates the charachteristic equation at a given point.

    Parameters:
        x ([float, float]):
            A list of 2 elements, making up the Re and Im parts of the complex eigenvalue to calculate char_eq.

    Returns:
        array[float, float]:
            An array of 2 elements, making up the Re and Im parts of the complex value of char_eq at the given x.
    """
    import numpy as np

    par = args[0]

    # Extract real and imaginary parts from the input tuple
    x_real, x_imag = x[0], x[1]

    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])

    # Create complex numbers element-wise from real and imaginary parts
    l = x_real + 1j * x_imag

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)

    is_zero = np.isclose(p_sqrt, 0, atol=1e-8)
    not_zero = ~is_zero
    # Create y array filled with NaN
    y = np.empty_like(l, dtype=np.complex128)

    # Calculate y element-wise based on the condition
    y[is_zero] = (
        np.exp(l[is_zero] * t + v / 2 / D) * (D + 2 * (k-l[is_zero]))
        + v * (1 - R * np.exp(v / D))
    )

    y[not_zero] = (
        np.exp(l[not_zero] * t + v / 2 / D) * np.sinh(p_sqrt[not_zero] / 2 / D) * (v**2 + 2 * D**2)
        + v * p_sqrt[not_zero] * (np.cosh(p_sqrt[not_zero] / 2 / D) - R * np.exp(v / D))
    )

    Y = np.array([np.real(y), np.imag(y)])


    return Y