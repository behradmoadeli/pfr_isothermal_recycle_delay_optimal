import numpy as np

y = np.array([3+1j,-2])
y_star = np.array([-3,2])

# y_sum = np.dot(y, y.conjugate()) + np.dot(y_star, y_star.conjugate())
y_sum = y*y.conjugate() + y_star*y_star.conjugate()
# y_sum = y**2 + y_star**2


print(y_sum)