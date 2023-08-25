def significant_digits(x, n):
    import numpy as np
    if x == 0:
        return 0
    return n - int(np.floor(np.log10(abs(x))))