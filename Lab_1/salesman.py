#!/usr/bin/python3

import sys
import math
import copy
import time
import random
from collections import deque


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
        # return f'{self.coords}'
        return f'{self.num}, coordinates: {self.coords}, neighbors: {self.neighbors}'


def calc_distance_S(A: list, B: list) -> float:
    distance_2 = 0
    for coord in range(len(A)):
        distance_2 += (A[coord] - B[coord]) ** 2
    return round(math.sqrt(distance_2), 2)

def calc_distance_NS(A: list, B: list) -> float:
    distance_2 = 0
    for coord in range(len(A)):
        distance_2 += (A[coord] - B[coord]) ** 2
    distance = math.sqrt(distance_2)
    if A[2] == B[2]: return round(distance, 2)
    distance += distance*0.1 if A[2] < B[2] else distance*-0.1
    return round(distance, 2)


def generate_cities(number_of_cities: int) -> list:
    cities = []
    cities_numbers_list = []
    for city in range(number_of_cities):
        cities.append(City(random.randint(-100, 100),
                      random.randint(-100, 100), random.randint(0, 50), city, number_of_cities))
        cities_numbers_list.append(city)
    return cities, cities_numbers_list


def create_distance_matrix(list_of_cities: list, calculate_distance_func) -> list:
    distance_matrix = []
    for city in range(len(list_of_cities)):
        distance_matrix.append([])
        for neighbor in range(len(list_of_cities)):
            distance = calculate_distance_func(
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


def calculate_cost(path: list, cities: list, calc_distance_function) -> float:
    cost = 0
    for number_of_city in range(len(path) - 1):
        cost += calc_distance_function(cities[path[number_of_city]].coords,
                              cities[path[number_of_city + 1]].coords)
    return cost


def DFS(visited_cities: list, cities_numbers_list: list, cities: list, current_cost: float, best_path: list, lowest_cost: float, calculate_distance_func):
    if len(visited_cities) == len(cities_numbers_list):
        final_cost = current_cost + \
            calculate_distance_func(cities[visited_cities[-1]].coords, cities[0].coords)
        if final_cost < lowest_cost:
            lowest_cost = final_cost
            best_path = visited_cities
            best_path.append(0)
        return best_path, lowest_cost
    else:
        for neighbor in cities[visited_cities[-1]].neighbors:
            if neighbor not in visited_cities and current_cost < lowest_cost:
                new_cost = current_cost + calculate_distance_func(
                    cities[visited_cities[-1]].coords, cities[neighbor].coords)
                visited_cities.append(neighbor)
                best_path, lowest_cost = DFS(
                    visited_cities[:], cities_numbers_list, cities, new_cost, best_path, lowest_cost, calculate_distance_func)
                visited_cities.remove(visited_cities[-1])
        return best_path, lowest_cost


def BFS(cities: list, calc_distance_function) -> None:

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
            cost = calculate_cost(path, cities, calc_distance_function)
            if cost < lowest_cost:
                best_path = path[:]
                lowest_cost = cost
    return best_path, lowest_cost


def nearest_neighbor(visited_cities: list, cities_numbers_list: list, cities: list, current_cost: float, distances, calc_distance_func):
    if len(visited_cities) == len(cities_numbers_list):
        final_path = visited_cities
        final_cost = current_cost + \
            calc_distance_func(cities[visited_cities[-1]].coords, cities[0].coords)
        final_path.append(0)
        return final_path, final_cost
    else:
        for city in visited_cities:
            distances[visited_cities[-1]][city] = float('inf')
        next_city = distances[visited_cities[-1]
                              ].index(min(distances[visited_cities[-1]]))
        new_cost = current_cost + calc_distance_func(
            cities[visited_cities[-1]].coords, cities[next_city].coords)
        visited_cities.append(next_city)
        final_path, final_cost = nearest_neighbor(
            visited_cities[:], cities_numbers_list, cities, new_cost, distances, calc_distance_func)
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
        min_expected if min_expected != float('inf') else 0
    return expected_cost

def heuristic_mean_IAD(visited: list, candidate_city: int, distances: list):
    distance_total = 0.0
    count = 0
    visited.append(candidate_city)
    for city in range(len(distances)):
        if city in visited:
            for neighbor in range(len(distances)):
                distances[city][neighbor] = float('inf')
                distances[neighbor][city] = float('inf')
    
    for city in range(len(distances)):
        for neighbor in range(len(distances)):
            if distances[city][neighbor] < float('inf'):
                distance_total += distances[city][neighbor]
                count += 1
    if count == 0: return 0
    return distance_total/count


def a_star(visited_cities: list, number_of_cities: int, cities: list, distances: list, heuristic, calc_distance_func):
    if len(visited_cities) == number_of_cities:
        final_path = visited_cities
        final_path.append(0)
        final_cost = calculate_cost(final_path, cities, calc_distance_func)
        return final_path, final_cost
    else:
        min_expected_cost = float('inf')
        next_city = 69
        for city in cities[visited_cities[-1]].neighbors:
            if city not in visited_cities:
                fixed_cost = calc_distance_func(
                    cities[visited_cities[-1]].coords, cities[city].coords)
                expected_cost = fixed_cost + \
                    heuristic(visited_cities[:], city,
                              copy.deepcopy(distances))
                if expected_cost <= min_expected_cost:
                    min_expected_cost = expected_cost
                    next_city = city
        visited_cities.append(next_city)
        final_path, final_cost = a_star(
            visited_cities[:], number_of_cities, cities, distances, heuristic, calc_distance_func)
        return final_path, final_cost


def a_star_2(start_path, number_of_cities: int, cities: list, distances: list, heuristic, calc_distance_func):
    priority_list = [(start_path, 0)]
    
    while len(priority_list[0][0]) != number_of_cities:
        path_to_check = priority_list.pop(0)
        for city in cities[path_to_check[0][-1]].neighbors:
            if city not in path_to_check[0]: 
                new_path = path_to_check[0][:]
                new_path.append(city)
                fixed_cost = calculate_cost(new_path, cities, calc_distance_func)
                expected_cost = fixed_cost + \
                    heuristic(path_to_check[0][:], city,
                                copy.deepcopy(distances))
                priority_list.append((new_path[:], expected_cost))
        priority_list.sort(key = lambda x: float(x[1]))
    return priority_list[0][0]
    


# def dijkstra(cities, cities_numbers_list, distance_matrix):
#     visited = [0]
#     path = [0]
#     not_visited = cities_numbers_list[1:]
#     print(not_visited)
#     predicted_distances = [{'shortest_distance': float('inf'), 'previous_city': None} for _ in cities_numbers_list]
#     predicted_distances[0]['shortest_distance'] = 0
#     predicted_distances[0]['previous_city'] = 'start'
#     print(predicted_distances)
    
def result(name, path, cost, time):
    print(f'{name}: best path: {path}, lowest cost: {cost:.2f}, duration: {time:.2f}s')


def all_conected(cities, cities_numbers_list, distance_matrix, calc_distance):
    print(f'Cities list: {cities_numbers_list}')
    DFS_start_time = Timer.start()
    DFS_best_path_final, DFS_lowest_cost = DFS(
        [0], cities_numbers_list, cities, 0, [], float('inf'), calc_distance)
    DFS_duration = Timer.stop(DFS_start_time)
    result('DFS', DFS_best_path_final, DFS_lowest_cost, DFS_duration)

    BFS_start_time = Timer.start()
    BFS_best_path, BFS_cost = BFS(cities, calc_distance_function=calc_distance)
    BFS_duration = Timer.stop(BFS_start_time)
    result('BFS', BFS_best_path, BFS_cost, BFS_duration)

    NN_start_time = Timer.start()
    NN_best_path, NN_cost = nearest_neighbor(
        [0], cities_numbers_list, cities, 0, copy.deepcopy(distance_matrix), calc_distance_func=calc_distance)
    NN_duration = Timer.stop(NN_start_time)
    result('NN', NN_best_path, NN_cost, NN_duration)

    # dijkstra_start_time = Timer.start()
    # dijkstra(cities, cities_numbers_list, distance_matrix)
    # dijkstra_best_path, dijkstra_cost = dijkstra(cities, cities_numbers_list, distance_matrix)
    # dijkstra_duration = Timer.stop(dijkstra_start_time)
    # result('Dijkstra', dijkstra_best_path, dijkstra_cost, dijkstra_duration)

    A_star_start_time = Timer.start()
    A_star_best_path, A_star_cost = a_star(
        [0], len(cities), cities, distance_matrix, heuristic_min_distance_AD, calc_distance_func=calc_distance)
    A_star_duration = Timer.stop(A_star_start_time)
    result('A star', A_star_best_path, A_star_cost, A_star_duration)
    
    A_star_start_time_IAD = Timer.start()
    A_star_best_path_IAD, A_star_cost_IAD = a_star(
        [0], len(cities), cities, distance_matrix, heuristic_mean_IAD, calc_distance_func=calc_distance)
    A_star_duration_IAD = Timer.stop(A_star_start_time_IAD)
    result('A star IAD', A_star_best_path_IAD, A_star_cost_IAD, A_star_duration_IAD)
    
    A_star_start_time = Timer.start()
    A_star_2_best_path = a_star_2([0], len(cities), cities, distance_matrix, heuristic_min_distance_AD, calc_distance )
    A_star_2_best_path.append(0)
    A_star_cost = calculate_cost(A_star_2_best_path, cities, calc_distance)
    A_star_duration = Timer.stop(A_star_start_time)
    result('A star 2 AD', A_star_2_best_path, A_star_cost, A_star_duration)
    
    A_star_start_time = Timer.start()
    A_star_2_best_path = a_star_2([0], len(cities), cities, distance_matrix, heuristic_mean_IAD, calc_distance )
    A_star_2_best_path.append(0)
    A_star_cost = calculate_cost(A_star_2_best_path, cities, calc_distance)
    A_star_duration = Timer.stop(A_star_start_time)
    result('A star 2 IAD', A_star_2_best_path, A_star_cost, A_star_duration)

    cities_numbers_list.append(0)
    random_cost = calculate_cost(cities_numbers_list, cities, calc_distance_function=calc_distance)
    result('Random path', cities_numbers_list, random_cost, 0)
    cities_numbers_list.pop()

def destroyed_roads(cities, cities_numbers_list, distance_matrix, calc_distance_func):
    DFS_start_time = Timer.start()
    DFS_best_path_final, DFS_lowest_cost = DFS(
        [0], cities_numbers_list, cities, 0, [], float('inf'), calc_distance_func)
    DFS_duration = Timer.stop(DFS_start_time)
    result('DFS destroyed roads:', DFS_best_path_final,
           DFS_lowest_cost, DFS_duration)

    BFS_start_time = Timer.start()
    BFS_best_path, BFS_cost = BFS(cities, calc_distance_func)
    BFS_duration = Timer.stop(BFS_start_time)
    result('BFS destroyed roads', BFS_best_path, BFS_cost, BFS_duration)

    NN_start_time = Timer.start()
    NN_best_path, NN_cost = nearest_neighbor(
        [0], cities_numbers_list, cities, 0, copy.deepcopy(distance_matrix), calc_distance_func)
    NN_duration = Timer.stop(NN_start_time)
    result('NN destroyed roads:', NN_best_path, NN_cost, NN_duration)
    
    A_star_start_time = Timer.start()
    A_star_best_path, A_star_cost = a_star(
        [0], len(cities), cities, distance_matrix, heuristic_min_distance_AD, calc_distance_func=calc_distance_func)
    A_star_duration = Timer.stop(A_star_start_time)
    result('A star', A_star_best_path, A_star_cost, A_star_duration)
    
    A_star_start_time_IAD = Timer.start()
    A_star_best_path_IAD, A_star_cost_IAD = a_star(
        [0], len(cities), cities, distance_matrix, heuristic_mean_IAD, calc_distance_func)
    A_star_duration_IAD = Timer.stop(A_star_start_time_IAD)
    result('A star IAD', A_star_best_path_IAD, A_star_cost_IAD, A_star_duration_IAD)
    
    A_star_start_time = Timer.start()
    A_star_2_best_path = a_star_2([0], len(cities), cities, distance_matrix, heuristic_min_distance_AD, calc_distance_func )
    A_star_2_best_path.append(0)
    A_star_cost = calculate_cost(A_star_2_best_path, cities, calc_distance_func)
    A_star_duration = Timer.stop(A_star_start_time)
    result('A star 2 AD', A_star_2_best_path, A_star_cost, A_star_duration)
    
    A_star_start_time = Timer.start()
    A_star_2_best_path = a_star_2([0], len(cities), cities, distance_matrix, heuristic_mean_IAD, calc_distance_func )
    A_star_2_best_path.append(0)
    A_star_cost = calculate_cost(A_star_2_best_path, cities, calc_distance_func)
    A_star_duration = Timer.stop(A_star_start_time)
    result('A star 2 IAD', A_star_2_best_path, A_star_cost, A_star_duration)
    
    # dijkstra_start_time = Timer.start()
    # dijkstra_best_path, dijkstra_cost = dijkstra()
    # dijkstra_duration = Timer.stop(dijkstra_start_time)
    # result('Dijkstra destroyed roads:', dijkstra_best_path, dijkstra_cost, dijkstra_duration)
    # print('miasta', cities, cities_numbers_list)
    # print('distance', distance_matrix)


if __name__ == '__main__':
    number_of_cities = int(sys.argv[1])
    cities, cities_numbers_list = generate_cities(number_of_cities)
    distance_matrix = create_distance_matrix(cities, calculate_distance_func=calc_distance_S)
    distance_matrix_NS = create_distance_matrix(cities, calculate_distance_func=calc_distance_NS)
    destroyed_distance_matrix, destroyed_cities = destroy_roads(
        copy.deepcopy(distance_matrix), copy.deepcopy(cities))
    destroyed_distance_matrix_NS, destroyed_cities_NS = destroy_roads(
        copy.deepcopy(distance_matrix_NS), copy.deepcopy(cities))
    print(f'Coordinates of cities: {cities}')
    print('ALL CONNECTED, SYMETRIC:')
    all_conected(cities, cities_numbers_list, distance_matrix, calc_distance_S)
    print('ALL CONNECTED, NON SYMETRIC:')
    all_conected(cities, cities_numbers_list, distance_matrix, calc_distance_NS)
    print('NOT ALL CONNECTED, SYMETRIC:')
    destroyed_roads(destroyed_cities, cities_numbers_list,
                    destroyed_distance_matrix, calc_distance_S)
    print('NOT ALL CONNECTED, NON SYMETRIC:')
    destroyed_roads(destroyed_cities_NS, cities_numbers_list,
                    destroyed_distance_matrix_NS, calc_distance_NS)
