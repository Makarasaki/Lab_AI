#!/usr/bin/python3

import math
import random
import numpy as np
import matplotlib.pyplot as plt


# def f(x, y):
#     "Objective function"
#     return ((1.5 - x - x*y) ** 2) + ((2.25 - x + ((x*y)**2))**2) + ((2.625 - x + ((x*y)**3))**2)


# # Contour plot: With the global minimum showed as "X" on the plot
# x, y = np.array(np.meshgrid(np.linspace(-4.5, 4.5, 1000),
#                 np.linspace(-4.5, 4.5, 1000)))
# print('x', x)
# print('y', y)
# # x = random.sample(range(-5, 5), 10)
# # y = random.sample(range(-5, 5), 10)
# z = f(x, y)
# print('z', z)
# x_min = x.ravel()[z.argmin()]
# print('x min', x_min)
# y_min = y.ravel()[z.argmin()]
# plt.figure(figsize=(8, 6))
# plt.imshow(z, extent=[-4.5, 5, -4.5, 5],
#            origin='lower', cmap='viridis', alpha=0.5)
# plt.colorbar()
# plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")
# contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
# plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
# plt.show()
# plt.savefig('myfilename.png', dpi=100)
# print('helo')

FIRST_DAY_DISTANCE = 2
DISTANCE_DECREASE_RATE = 0.9
NUMBER_OF_DAYS = 10
SQRT_NUMBER_OF_SEARCHERS = 10
PB_WEIGHT = 1
GB_WEIGHT = 1


class Searcher:
    def __init__(self, coords) -> None:
        self.coords = coords
        self.personal_best = 0
        self.pb_coords = [2, 3]

    def new_localization(self, gb_coords, path_len):
        # pb_weight = 1
        # gb_weight = 1
        pb_vector = [self.pb_coords[0] - self.coords[0],
                     self.pb_coords[1] - self.coords[1]]
        pb_v_len = math.sqrt((pb_vector[0]**2) + (pb_vector[1]**2))
        pb_versor = np.array(pb_vector)/pb_v_len

        gb_vector = [gb_coords[0] - self.coords[0],
                     gb_coords[1] - self.coords[1]]
        gb_v_len = math.sqrt((gb_vector[0]**2) + (gb_vector[1]**2))
        gb_versor = np.array(gb_vector)/gb_v_len

        resultant_vector = [pb_versor[0]*PB_WEIGHT+gb_versor[0] *
                            GB_WEIGHT, pb_versor[1]*PB_WEIGHT+gb_versor[1]*GB_WEIGHT]
        resultant_vector_len = math.sqrt(
            (resultant_vector[0]**2) + (resultant_vector[1]**2))
        resultant_versor = np.array(resultant_vector)/resultant_vector_len

        self.coords = [self.coords[0] + (resultant_versor[0] * path_len),
                       self.coords[1] + (resultant_versor[1] * path_len)]


class Global_best:
    def __init__(self) -> None:
        self.value = None
        self.coords = [5, 6]


if __name__ == '__main__':
    searcher1 = Searcher([0, 0])
    global_best = Global_best()
    searcher1.new_localization(global_best.coords, 10)
    print(searcher1.coords)
