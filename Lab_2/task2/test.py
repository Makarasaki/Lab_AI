#!/usr/bin/python3
# import math
# import random
import numpy as np
import matplotlib.pyplot as plt

X_RANGE_MIN = -20.5
X_RANGE_MAX = 20.5
Y_RANGE_MIN = -20.5
Y_RANGE_MAX = 20.5

def f(x, y):
    "Objective function"
    return ((1.5 - x - x*y) ** 2) + ((2.25 - x + ((x*y)**2))**2) + ((2.625 - x + ((x*y)**3))**2)


# Contour plot: With the global minimum showed as "X" on the plot
x, y = np.array(np.meshgrid(np.linspace(Y_RANGE_MIN, X_RANGE_MAX, 1000),
                np.linspace(Y_RANGE_MIN, Y_RANGE_MAX, 1000)))
print('x', x)
print('y', y)
# x = random.sample(range(-5, 5), 10)
# y = random.sample(range(-5, 5), 10)
z = f(x, y)
print('z', z)
x_min = x.ravel()[z.argmin()]
print('x min', x_min)
y_min = y.ravel()[z.argmin()]
plt.figure(figsize=(8, 6))
plt.imshow(z, extent=[X_RANGE_MIN, X_RANGE_MAX, Y_RANGE_MIN, Y_RANGE_MAX],
           origin='lower', cmap='viridis', alpha=0.5)
plt.colorbar()
plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")
contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
plt.show()
plt.savefig('myfilename.png', dpi=100)
print('helo')
print('minimum value', f(x_min, y_min))
