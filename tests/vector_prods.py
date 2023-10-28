import numpy as np

a = np.array([1, 2])
b = np.array([4, 5])

inner_product = np.dot(a, b)

p = np.arange(1, 5).reshape(2,2)


print(np.dot(b,p[:,0]) * np.dot(b,p[:,1]))

s_1 = 0
s_2 = 0

for i in range(2):
    s_1 += b[i] * p[i,0]
    s_2 += b[i] * p[i,1]

print(s_1 * s_2)