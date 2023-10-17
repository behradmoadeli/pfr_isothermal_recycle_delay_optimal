from scipy.integrate import quad, trapz
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

par = {'k': 1.5, 'D': 0.05, 'v': 1, 'tau': 0.8, 'R': 0.6, 'label': 'default'}
l = [0.57438205, 0.22125264-3.35991407*1j]
(k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])

# print(complex(*quad(eig_fun_mul_1,0,1,args=(par, l),complex_func=True)))

# x=0
# print(eig_fun_adj_2(x, par, l[0])-eig_fun_adj_1(x, par, l[0]))
# print(eig_fun_adj_1_prime(x, par, l[0]))
# print(eig_fun_adj_1_prime(x, par, l[1]))

# x=1
# print(eig_fun_2(x, par, l[1])-eig_fun_1(x, par, l[1]))
# print(eig_fun_1_prime(x, par, l[0]))
# print(eig_fun_1_prime(x, par, l[1]))

# print(D * eig_fun_adj_1_prime(1, par, l[1]) + v * eig_fun_adj_1(1, par, l[1]) - R * v * eig_fun_adj_2(1, par, l[1]))
# print(D * eig_fun_1_prime(0, par, l[1]) - v * eig_fun_1(0, par, l[1]) + R * v * eig_fun_2(0, par, l[1]))

# def my_fun(l):
#     l = l[0]+l[1]*1j
#     y = D * eig_fun_adj_1_prime(1, par, l) + v * eig_fun_adj_1(1, par, l) - R * v * eig_fun_adj_2(1, par, l)
#     return y.real, y.imag


# print(fsolve(my_fun,[0.5,0]))

# x = np.linspace(0.2,0.25,1000)
# y = []
# for i in x:
#     y.append(my_fun([i, 3.35991407])[0])

# plt.plot(x,np.real(y))
# plt.show()

# x = np.linspace(0,1,10000)
# y=[]

# phi = []
# psi = []
# phi_star = []
# psi_star = []

# for i in x:
#     phi.append(eig_fun_1(i, par, l[0]))
#     psi.append(eig_fun_2(i, par, l[0]))
#     phi_star.append(eig_fun_adj_1(i, par, l[0]))
#     psi_star.append(eig_fun_adj_2(i, par, l[0]))    
    # y.append(eig_fun_mul_1(i, par, l))

# plt.plot(x,phi_star)
# plt.show()
# I = trapz(y,x)
# print(I)

print(complex(*quad(eig_fun_mul_1,0,1,args=(par, l),complex_func=True)))
