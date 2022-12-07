#!/usr/bin/python3

import sys
import math
import copy
import time
import random
# import numpy as np
# import networkx as nx
from collections import deque
# import matplotlib.pyplot as plt


class Timer:
    def start():
        return time.perf_counter()

    def stop(start_time):
        return time.perf_counter() - start_time


class City:
    def __init__(self, x: int, y: int, z: int, city_num: int, number_of_cities: int) -> None:
        self.coords = [x, y, z]
        self.neighbors = list(range(0, number_of_cities))
        self.neighbors.remove(city_num)
        self.num = city_num

    def __repr__(self) -> str:
        return f'{self.num}, coordinates: {self.coords}, neighbors: {self.neighbors}'


# class Result():
#     def __init__(self):
#         self.lowest_cost = 99999999
#         self.path = []

#     def __repr__(self) -> str:
#         return f'cost: {self.lowest_cost}, path: {self.path}'


def calc_distance(A: list, B: list) -> float:
    distance_2 = 0
    for coord in range(len(A)):
        distance_2 += (A[coord] - B[coord]) ** 2
    return round(math.sqrt(distance_2), 2)


def generate_cities(number_of_cities: int) -> list:
    cities = []
    cities_numbers_list = []
    for city in range(number_of_cities):
        cities.append(City(random.randint(-100, 100),
                      random.randint(-100, 100), random.randint(0, 50), city, number_of_cities))
        cities_numbers_list.append(city)
    return cities, cities_numbers_list


def create_distance_matrix(list_of_cities: list) -> list:
    distance_matrix = []
    for city in range(len(list_of_cities)):
        distance_matrix.append([])
        for neighbor in range(len(list_of_cities)):
            distance = calc_distance(
                list_of_cities[city].coords, list_of_cities[neighbor].coords)
            distance = distance if distance != 0 else float('inf')
            distance_matrix[city].append(distance)
    return distance_matrix


def destroy_roads(distance_matrix: list, cities: list) -> list:
    number_of_cities = len(distance_matrix)
    number_of_roads_to_destroy = int(len(distance_matrix)*0.2)
    while number_of_roads_to_destroy > 0:
        road_to_destroy_city_A = random.randint(0, number_of_cities - 1)
        road_to_destroy_city_B = random.randint(0, number_of_cities - 1)
        if distance_matrix[road_to_destroy_city_A][road_to_destroy_city_B] != float('inf') and distance_matrix[road_to_destroy_city_B][road_to_destroy_city_A] != float('inf'):
            distance_matrix[road_to_destroy_city_A][road_to_destroy_city_B] = float(
                'inf')
            distance_matrix[road_to_destroy_city_B][road_to_destroy_city_A] = float(
                'inf')
            cities[road_to_destroy_city_A].neighbors.remove(
                road_to_destroy_city_B)
            cities[road_to_destroy_city_B].neighbors.remove(
                road_to_destroy_city_A)
            number_of_roads_to_destroy -= 1
    return distance_matrix, cities


def calculate_cost(path: list, cities: list) -> float:
    cost = 0
    for number_of_city in range(len(path) - 1):
        cost += calc_distance(cities[path[number_of_city]].coords,
                              cities[path[number_of_city + 1]].coords)
    return cost


def DFS(visited_cities: list, cities_numbers_list: list, cities: list, current_cost: float, best_path: list, lowest_cost: float):
    if len(visited_cities) == len(cities_numbers_list):
        final_cost = current_cost + \
            calc_distance(cities[visited_cities[-1]].coords, cities[0].coords)
        if final_cost < lowest_cost:
            lowest_cost = final_cost
            best_path = visited_cities
            best_path.append(0)
        return best_path, lowest_cost
    else:
        for neighbor in cities[visited_cities[-1]].neighbors:
            if neighbor not in visited_cities and current_cost < lowest_cost:
                new_cost = current_cost + calc_distance(
                    cities[visited_cities[-1]].coords, cities[neighbor].coords)
                visited_cities.append(neighbor)
                best_path, lowest_cost = DFS(
                    visited_cities[:], cities_numbers_list, cities, new_cost, best_path, lowest_cost)
                visited_cities.remove(visited_cities[-1])
        return best_path, lowest_cost


def BFS(cities: list) -> None:

    def if_visited(city: int, path: list) -> int:
        for visited_city in range(len(path)):
            if (path[visited_city] == city):
                return 0
        return 1

    roads = []
    path = []
    all_paths = []
    best_path = []
    lowest_cost = float('inf')
    queue = deque()
    path.append(0)
    queue.append(path[:])

    for city in cities:
        roads.append(city.neighbors)

    while queue:
        path = queue.popleft()
        last = path[-1]
        if (len(path) == len(roads)):
            path.append(0)
            all_paths.append(path)

        for neighbor in range(len(roads[last])):
            if (if_visited(roads[last][neighbor], path)):
                newpath = path.copy()
                newpath.append(roads[last][neighbor])
                queue.append(newpath)

    for path in all_paths:
        if len(path) == len(cities) + 1:
            cost = calculate_cost(path, cities)
            if cost < lowest_cost:
                best_path = path[:]
                lowest_cost = cost
    return best_path, lowest_cost


def nearest_neighbor(visited_cities: list, cities_numbers_list: list, cities: list, current_cost: float, distances):
    if len(visited_cities) == len(cities_numbers_list):
        final_path = visited_cities
        final_cost = current_cost + \
            calc_distance(cities[visited_cities[-1]].coords, cities[0].coords)
        final_path.append(0)
        return final_path, final_cost
    else:
        for city in visited_cities:
            distances[visited_cities[-1]][city] = float('inf')
        next_city = distances[visited_cities[-1]
                              ].index(min(distances[visited_cities[-1]]))
        new_cost = current_cost + calc_distance(
            cities[visited_cities[-1]].coords, cities[next_city].coords)
        visited_cities.append(next_city)
        final_path, final_cost = nearest_neighbor(
            visited_cities[:], cities_numbers_list, cities, new_cost, distances)
        return final_path, final_cost


def heuristic_min_distance_AD(visited: list, candidate_city: int, distances: list):
    min_expected = float('inf')
    visited.append(candidate_city)
    for city in range(len(distances)):
        if city in visited:
            for neighbor in range(len(distances)):
                distances[city][neighbor] = float('inf')
                distances[neighbor][city] = float('inf')

    for city in range(len(distances)):
        for neighbor in range(len(distances)):
            if distances[city][neighbor] < min_expected:
                min_expected = distances[city][neighbor]

    expected_cost = (len(distances) - len(visited) - 1) * \
        min_expected if min_expected != float('inf') else len(distances) * 1000
    return expected_cost


def a_star(visited_cities: list, number_of_cities: int, cities: list, distances: list, heuristic):
    if len(visited_cities) == number_of_cities:
        final_path = visited_cities
        final_path.append(0)
        final_cost = calculate_cost(final_path, cities)
        return final_path, final_cost
    else:
        min_expected_cost = float('inf')
        next_city = 10
        for city in cities[visited_cities[-1]].neighbors:
            if city not in visited_cities:
                fixed_cost = calc_distance(
                    cities[visited_cities[-1]].coords, cities[city].coords)
                expected_cost = fixed_cost + \
                    heuristic(visited_cities[:], city,
                              copy.deepcopy(distances))
                if expected_cost <= min_expected_cost:
                    min_expected_cost = expected_cost
                    next_city = city
        visited_cities.append(next_city)
        final_path, final_cost = a_star(
            visited_cities[:], number_of_cities, cities, distances, heuristic)
        return final_path, final_cost


def dijkstra():
    pass


def result(name, path, cost, time):
    print(f'{name}: best path: {path}, lowest cost: {cost:.2f}, duration: {time:.2f}s')


def all_conected(cities, cities_numbers_list, distance_matrix):
    DFS_start_time = Timer.start()
    DFS_best_path_final, DFS_lowest_cost = DFS(
        [0], cities_numbers_list, cities, 0, [], float('inf'))
    DFS_duration = Timer.stop(DFS_start_time)
    result('DFS', DFS_best_path_final, DFS_lowest_cost, DFS_duration)

    BFS_start_time = Timer.start()
    BFS_best_path, BFS_cost = BFS(cities)
    BFS_duration = Timer.stop(BFS_start_time)
    result('BFS', BFS_best_path, BFS_cost, BFS_duration)

    NN_start_time = Timer.start()
    NN_best_path, NN_cost = nearest_neighbor(
        [0], cities_numbers_list, cities, 0, copy.deepcopy(distance_matrix))
    NN_duration = Timer.stop(NN_start_time)
    result('NN', NN_best_path, NN_cost, NN_duration)

    dijkstra_start_time = Timer.start()
    # dijkstra_best_path, dijkstra_cost = dijkstra()
    dijkstra_duration = Timer.stop(dijkstra_start_time)
    # result('Dijkstra', dijkstra_best_path, dijkstra_cost, dijkstra_duration)

    A_star_start_time = Timer.start()
    A_star_best_path, A_star_cost = a_star(
        [0], len(cities), cities, distance_matrix, heuristic_min_distance_AD)
    A_star_duration = Timer.stop(A_star_start_time)
    result('A star', A_star_best_path, A_star_cost, A_star_duration)

    cities_numbers_list.append(0)
    random_cost = calculate_cost(cities_numbers_list, cities)
    result('Random path', cities_numbers_list, random_cost, 0)
    cities_numbers_list.remove(0)
    # print('miasta', cities, cities_numbers_list)
    # print('distance', distance_matrix)

    # heuristic_min_distance_AD([0, 3], distance_matrix[:])


def destroyed_roads(cities, cities_numbers_list, distance_matrix):
    DFS_start_time = Timer.start()
    DFS_best_path_final, DFS_lowest_cost = DFS(
        [0], cities_numbers_list, cities, 0, [], float('inf'))
    DFS_duration = Timer.stop(DFS_start_time)
    result('DFS destroyed roads:', DFS_best_path_final,
           DFS_lowest_cost, DFS_duration)

    BFS_start_time = Timer.start()
    BFS_best_path, BFS_cost = BFS(cities)
    BFS_duration = Timer.stop(BFS_start_time)
    result('BFS destroyed roads', BFS_best_path, BFS_cost, BFS_duration)

    NN_start_time = Timer.start()
    NN_best_path, NN_cost = nearest_neighbor(
        [0], cities_numbers_list, cities, 0, copy.deepcopy(distance_matrix))
    NN_duration = Timer.stop(NN_start_time)
    result('NN destroyed roads:', NN_best_path, NN_cost, NN_duration)

    dijkstra_start_time = Timer.start()
    # dijkstra_best_path, dijkstra_cost = dijkstra()
    dijkstra_duration = Timer.stop(dijkstra_start_time)
    # result('Dijkstra destroyed roads:', dijkstra_best_path, dijkstra_cost, dijkstra_duration)
    # print('miasta', cities, cities_numbers_list)
    # print('distance', distance_matrix)


def draw_graph(cities):
    roads = []
    locations = []

    for city in cities:
        locations.append((city.coords[0], city.coords[1]))
        for road in city.neighbors:
            roads.append((city.num, road))
    G = nx.DiGraph(directed=True)
    G.add_edges_from(roads)

    options = {
        'node_color': 'green',
        'node_size': 200,
        'width': 1,
        'arrowstyle': '->',
        'arrowsize': 20,
    }

    val_map = {'1': 1.0, '5': 0.5714285714285714, '6': 0.0}

    values = [val_map.get(node, 0.25) for node in G.nodes()]
    # generating pos dictionary

    # pos = {str(i):location for i, location in enumerate(locations)}
    pos = nx.spring_layout(G)
    print('POSITION', pos)

    # drawing graph, with positions included.
    nx.draw_networkx(G, pos=pos, arrows=True, **options)

    plt.show()


if __name__ == '__main__':
    number_of_cities = int(sys.argv[1])
    cities, cities_numbers_list = generate_cities(number_of_cities)
    distance_matrix = create_distance_matrix(cities)
    destroyed_distance_matrix, destroyed_cities = destroy_roads(
        copy.deepcopy(distance_matrix), copy.deepcopy(cities))
    # print(cities, destroyed_cities)

    all_conected(cities, cities_numbers_list, distance_matrix)

    # destroyed_roads(destroyed_cities, cities_numbers_list,
    #                 destroyed_distance_matrix)

    # draw_graph(cities)
    # draw_graph(destroyed_cities)
