from scipy.optimize import fsolve
import numpy as np

x0 = np.array([
    [1,2],
    [3,4],
    [5,6]
])

def myfun(m):
    x = m[0]
    y = m[1]
    r = []
    r.append(x**2 + 2*y)
    r.append(2*y - 3 + np.sin(x-y))
    return r

x = fsolve(myfun, x0)
