#!/usr/bin/python3

import copy
import random
import pandas as pd
from termcolor import colored

NUMBER_OF_COLUMNS = 99
NUMBER_OF_ROWS = 12
NUMBER_OF_RESOURCES = 10
JOBS_PER_PRODUCT = 10

NUMBER_OF_GENERATIONS = 100
POPULATION_SIZE = 50  # MUST BE EVEN
REJECTION_SIZE = 5
NUMBER_OF_MUTATIONS = 1


class Product:
    def __init__(self, tasks) -> None:
        self.tasks_left = tasks
        self.busy = False
        self.when_free = 0

    def __repr__(self) -> str:
        return f'tasks: {self.tasks_left}, when free: {self.when_free}'


class Resource:
    def __init__(self, number) -> None:
        self.busy = False
        self.when_free = 0
        self.number = number

    def __repr__(self) -> str:
        return f'Resource {self.number}: {self.when_free} {self.busy}'


class Schedule:
    def __init__(self, priority) -> None:
        self.priority = priority
        self.time = float('inf')

    def __repr__(self) -> str:
        return f'Time: {self.time} Priority: {self.priority}'


def upload_tasks():
    tasks_data = pd.read_excel("GA_task.xlsx")
    return tasks_data


def extract_products_instructions(tasks_data):
    products = []
    for column in range(0, NUMBER_OF_COLUMNS, 2):
        tasks = []
        for row in range(1, NUMBER_OF_ROWS):
            tasks.append([tasks_data.iloc[row, column],
                         tasks_data.iloc[row, column + 1]])
        products.append(Product(tasks))
    return products


def extract_resources_schedules(products):
    resources = []
    for _ in range(0, NUMBER_OF_RESOURCES):
        resources.append([])
    for product in products:
        for job in product:
            resources[job[0]-1].append([products.index(product)+1, job[1]])
    for list in resources:
        pass


def combine_dna(dna_1, dna_2, cycles):
    child_dna_1 = []
    child_dna_2 = []
    for index in range(len(dna_1)):
        child_dna_1.append(0)
        child_dna_2.append(0)
    even_distribution_counter = 0
    for cycle in cycles:
        for index in cycle:
            if (even_distribution_counter % 2) == 0:
                child_dna_1[index] = dna_1[index]
                child_dna_2[index] = dna_2[index]
            else:
                child_dna_1[index] = dna_2[index]
                child_dna_2[index] = dna_1[index]
        even_distribution_counter += 1
    # print("DZIECI: ", child_dna_1, child_dna_2)
    return child_dna_1, child_dna_2


def crossover(dna_1, dna_2):
    cycles = []
    for nucleotides in dna_1:
        if not any(dna_1.index(nucleotides) in nested_list for nested_list in cycles):
            wanted_nucleotide = nucleotides
            current_nucleotide_2 = 0
            current_nucleotide_1 = wanted_nucleotide
            current_cycle = []
            while current_nucleotide_2 != wanted_nucleotide:
                current_cycle.append(dna_1.index(current_nucleotide_1))
                current_nucleotide_2 = dna_2[dna_1.index(current_nucleotide_1)]
                current_nucleotide_1 = dna_1[dna_1.index(current_nucleotide_2)]
            cycles.append(current_cycle)
    return combine_dna(dna_1, dna_2, cycles)


def resources_busy(resources):
    for resource in resources:
        if resource.busy == True or resource.when_free == 0:
            return True
    return False


def calc_time(schedule, resources, products):
    time = 0
    while resources_busy(resources):
        for resource in resources:
            if resource.when_free <= time:
                resource.busy = False
            else:
                continue
            for product in schedule:
                product = products[product]
                if product.tasks_left == []:
                    continue
                if product.when_free <= time:
                    product.busy = False
                # print(resource.busy, product.busy, product.tasks_left[0][0], resource.number)
                if (resource.busy == False) and (product.busy == False) and (product.tasks_left[0][0] == resource.number):
                    resource.busy = True
                    resource.when_free = time + product.tasks_left[0][1]
                    product.busy = True
                    product.when_free = time + product.tasks_left[0][1]
                    del product.tasks_left[0]
        time += 1
    # print('CZAS: ',time)
    return time


def genetic_algotithm(products):
    resources = []
    for resource in range(JOBS_PER_PRODUCT):
        resources.append(Resource(resource + 1))

    population = []
    for schedule in range(POPULATION_SIZE):
        population.append(Schedule(random.sample(range(0, 50), 50)))
    previous_generation = copy.deepcopy(population)
    best_ever = copy.deepcopy(population[0])
    # calc_time(population[0], resources, products)
    for generation in range(NUMBER_OF_GENERATIONS):
        for schedule in population:
            schedule.time = calc_time(schedule.priority, copy.deepcopy(
                resources), copy.deepcopy(products))
        two_populations = previous_generation + population
        two_populations.sort(key=lambda x: x.time, reverse=False)
        # print('TWO_POPULATIONS: ', two_populations, len(two_populations))
        population = two_populations[:POPULATION_SIZE]
        # previous_generation = copy.deepcopy(population)
        print(
            f'BEST FROM GENERATION {generation}: {population[0]}, {len(population)}')
        if population[0].time < best_ever.time:
            best_ever = copy.deepcopy(population[0])
        # print(f'GENERATION {generation}, {population}')
        for change in range(REJECTION_SIZE):
            population[-1-change] = copy.deepcopy(population[change])
        # crossover
        pairing_list = random.sample(
            range(0, POPULATION_SIZE), POPULATION_SIZE)
        for pair in range(0, len(pairing_list), 2):
            population[pair].priority, population[pair + 1].priority = crossover(
                population[pair].priority, population[pair+1].priority)
            population[pair].time = float('inf')
            population[pair + 1].time = float('inf')

        for kid in population:
            for _ in range(NUMBER_OF_MUTATIONS):
                mutation_list = random.sample(range(0, 50), 2)
                nucleotide = kid.priority[mutation_list[0]]
                kid.priority[mutation_list[0]] = kid.priority[mutation_list[1]]
                kid.priority[mutation_list[1]] = nucleotide
    return best_ever


if __name__ == '__main__':
    tasks_data = upload_tasks()
    products = extract_products_instructions(tasks_data)
    print(colored(('BEST EVER: ', genetic_algotithm(products)), 'red'))
