from functions import *
import numpy as np
import matplotlib.pyplot as plt

default_pars = obtain_default_pars('/Users/behradmoadeli/Documents/PhD/behrads_papers/runs/run_5/pars_list.csv')

# # Adjust the range of X_r and X_i as needed
# X_r = np.linspace(-14.6,-14.4,100)
# X_i = np.linspace(116.5,116.7,100)
# x_r, x_i = np.meshgrid(X_r, X_i)

# # Calculate y (absolute value of complex result)
# y = char_eq((x_r, x_i), default_pars)
# Y = np.full_like(y[0], np.nan)

# for i, row in enumerate(y[0]):
#     for j, col in enumerate(row):
#         Y[i,j] = np.abs(complex(y[0,i,j], y[1,i,j]))

# Y[np.isinf(Y)] = np.max(Y[np.isfinite(Y)])

X_r = np.linspace(0,1,50)
X_i = np.linspace(0,1,50)
Y = np.array([np.linspace(1, 3, 50) for _ in X_r])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_r, X_i, Y)

# Adding labels
ax.set_xlabel('real')
ax.set_ylabel('imag')
ax.set_zlabel('char_eq')

# Adjusting the viewing angle
# ax.view_init(azim=45, elev=30)  # Change azim and elev to set the viewing angle

# Adding a color bar using the existing surface plot
# fig.colorbar(surf, ax=ax)

plt.show()