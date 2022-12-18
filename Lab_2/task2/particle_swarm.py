#!/usr/bin/python3

import math
# import random
import numpy as np
import matplotlib.pyplot as plt

FIRST_DAY_DISTANCE = 1
DISTANCE_DECREASE_RATE = 0.9
NUMBER_OF_DAYS = 12
SQRT_NUMBER_OF_SEARCHERS = 2
PB_WEIGHT = 1
GB_WEIGHT = 1.1
X_RANGE_MIN = -4.5
X_RANGE_MAX = 4.5
Y_RANGE_MIN = -4.5
Y_RANGE_MAX = 4.5


class Searcher:
    def __init__(self, coords: list) -> None:
        self.coords = coords
        self.personal_best = self.calc_value()
        self.pb_coords = coords

    # def __repr__(self) -> str:
    #     return f'coordinates: {self.coords}, personal best: {self.personal_best}, PB coordinates: {self.pb_coords}'

    def __repr__(self) -> str:
        return f'[{self.coords[0]:.2f}; {self.coords[1]:.2f}, {self.personal_best:.2f}'

    def calc_value(self):
        x = self.coords[0]
        y = self.coords[1]
        return ((1.5 - x - x*y) ** 2) + ((2.25 - x + ((x*y)**2))**2) + ((2.625 - x + ((x*y)**3))**2)

    def check_boundaries(self):
        for i in range(0, 1):
            if self.coords[i] > X_RANGE_MAX:
                self.coords[i] = X_RANGE_MAX
            if self.coords[i] < X_RANGE_MIN:
                self.coords[i] = X_RANGE_MIN

    def update_pb(self):
        current_value = self.calc_value()
        if current_value < self.personal_best:
            self.personal_best = current_value
            self.pb_coords = self.coords

    def calculate_versor(self, coords):
        vector = [coords[0] - self.coords[0],
                  coords[1] - self.coords[1]]
        v_len = math.sqrt((vector[0]**2) + (vector[1]**2))
        return np.array(vector)/v_len

    def new_localization(self, gb_coords, path_len):
        pb_versor = [0, 0]
        gb_versor = [0, 0]
        if gb_coords == self.coords and self.pb_coords == self.coords:
            return

        if self.pb_coords != self.coords:
            pb_versor = self.calculate_versor(self.pb_coords)

        if gb_coords != self.coords:
            gb_versor = self.calculate_versor(gb_coords)

        resultant_vector = [pb_versor[0]*PB_WEIGHT+gb_versor[0] *
                            GB_WEIGHT, pb_versor[1]*PB_WEIGHT+gb_versor[1]*GB_WEIGHT]
        resultant_vector_len = math.sqrt(
            (resultant_vector[0]**2) + (resultant_vector[1]**2))
        resultant_versor = np.array(resultant_vector)/resultant_vector_len

        self.coords = [self.coords[0] + (resultant_versor[0] * path_len),
                       self.coords[1] + (resultant_versor[1] * path_len)]
        self.check_boundaries()
        # Update Private Best
        self.update_pb()


class Global_best:
    def __init__(self, value: float, coords: list) -> None:
        self.value = value
        self.coords = coords

    def __repr__(self) -> str:
        return f'GLOBAL BEST: {self.coords}, {self.value}'

    def calc_value(self):
        x = self.coords[0]
        y = self.coords[1]
        return ((1.5 - x - x*y) ** 2) + ((2.25 - x + ((x*y)**2))**2) + ((2.625 - x + ((x*y)**3))**2)

    def update_best(self, searcher):
        self.value = searcher.personal_best
        self.coords = searcher.coords
        # self.value = self.calc_value()

def show_plot():
    def f(x, y):
        "Objective function"
        return ((1.5 - x - x*y) ** 2) + ((2.25 - x + ((x*y)**2))**2) + ((2.625 - x + ((x*y)**3))**2)
    x, y = np.array(np.meshgrid(np.linspace(Y_RANGE_MIN, X_RANGE_MAX, 1000),
                np.linspace(Y_RANGE_MIN, Y_RANGE_MAX, 1000)))
    z = f(x, y)
    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]
    plt.figure(figsize=(8, 6))
    plt.imshow(z, extent=[X_RANGE_MIN, X_RANGE_MAX, Y_RANGE_MIN, Y_RANGE_MAX],
            origin='lower', cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")
    contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    plt.show()
    # plt.savefig('myfilename.png', dpi=100)

def place_searchcers():
    searchers_list = []
    grid = np.linspace(X_RANGE_MIN, X_RANGE_MAX, SQRT_NUMBER_OF_SEARCHERS)
    for x in grid:
        for y in grid:
            searchers_list.append(Searcher([x, y]))
    return searchers_list


def find_current_best(searchers_list: list) -> object:
    global_best = Global_best(float('inf'), [0, 0])
    for searcher in searchers_list:
        if searcher.personal_best < global_best.value:
            global_best.update_best(searcher)
    return global_best


def particle_swarm(searchers_list: list):
    global_best = find_current_best(searchers_list)
    print(global_best)
    for day in range(0, NUMBER_OF_DAYS):
        for searcher in searchers_list:
            searcher.new_localization(
                global_best.coords, FIRST_DAY_DISTANCE*(DISTANCE_DECREASE_RATE**day))
            if searcher.personal_best < global_best.value:
                global_best.update_best(searcher)
                print(global_best)
    print(global_best)
    print(searchers_list)


if __name__ == '__main__':
    searchers_list = place_searchcers()
    print(searchers_list)
    particle_swarm(searchers_list)
    show_plot()
