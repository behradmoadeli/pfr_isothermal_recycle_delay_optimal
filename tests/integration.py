from scipy.integrate import quad
from functions import *


par = {'k': 1.5, 'D': 0.05, 'v': 1, 'tau': 0.8, 'R': 0.6, 'label': 'default'}
l = [0.5459, 0.1907-3.463*1j]

def eig_fun_mul_1(x, *args):
    
    par = args[0]
        
    if hasattr(args[1], '__iter__'):
        l = args[1]
    else:
        l = [args[1]]*2

    phi = eig_fun_1(x, par, l[0])
    psi = eig_fun_2(x, par, l[0])
    
    phi_star = eig_fun_adj_1(x, par, l[1])
    psi_star = eig_fun_adj_2(x, par, l[1])

    return phi * phi_star + psi * psi_star

print(complex(*quad(eig_fun_mul_1,0,1,args=(par, l),complex_func=True)))

x=0
print(eig_fun_adj_2(x, par, l[1]))
print(eig_fun_adj_1(x, par, l[1]))

x=1
print(eig_fun_2(x, par, l[0]))
print(eig_fun_1(x, par, l[0]))
