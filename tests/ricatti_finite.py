from scipy.optimize import fsolve
from functions import *
import numpy as np

default_pars = {'k': 1.5, 'D': 0.05, 'v': 1, 'tau': 0.8, 'R': 0.6, 'label': 'default'}
lambdas = [0.57438+0j, 0.22125 - 3.3599j, 0.22125 + 3.3599j]
normal_coefs = [-0.0298-0j, 0.0182-0.1027j, 0.0182+0.1027j]

p = np.ones((3,3), dtype=float) * 1000
p_0 = [*triu_to_flat(p)] * 2

p_sol_flat = fsolve(ricatti_finite, p_0, (default_pars, lambdas, normal_coefs))


slicer = int(len(p_sol_flat)/2)
p_sol_flat_real = p_sol_flat[:slicer]
p_sol_flat_imag = p_sol_flat[slicer:]
p_sol_flat_complex = p_sol_flat_real + p_sol_flat_imag * 1j

p_sol = triu_to_symm(flat_to_triu(p_sol_flat_complex))
    
print(p_sol)