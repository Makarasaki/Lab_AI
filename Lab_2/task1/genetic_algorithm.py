#!/usr/bin/python3

import random
import pandas as pd

NUMBER_OF_COLUMNS = 99
NUMBER_OF_ROWS = 12
NUMBER_OF_RESOURCES = 10


def upload_tasks():
    tasks_data = pd.read_excel("GA_task.xlsx")
    return tasks_data


def extract_products_instructions(tasks_data):
    products = []
    for column in range(0, NUMBER_OF_COLUMNS, 2):
        products.append([])
        for row in range(1, NUMBER_OF_ROWS):
            products[int(column/2)].append([tasks_data.iloc[row,
                                                            column], tasks_data.iloc[row, column + 1]])
    return products


def extract_resources_schedules(products):
    resources = []
    for _ in range(0, NUMBER_OF_RESOURCES):
        resources.append([])
    print(resources)
    for product in products:
        for job in product:
            print(job)
            print('product', product)
            resources[job[0]-1].append([products.index(product)+1, job[1]])
    # print('resources', resources)
    for list in resources:
        print('resource: ', list)


def calculate_time(resources):
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
                child_dna_1[index]=dna_2[index]
                child_dna_2[index]=dna_1[index]
        even_distribution_counter += 1
    return child_dna_1, child_dna_2

def crossover(dna_1, dna_2):
    cycles = []
    for nucleotides in dna_1:
        if not any(dna_1.index(nucleotides) in nested_list  for nested_list  in cycles):
            wanted_nucleotide = nucleotides
            current_nucleotide_2 = 0
            current_nucleotide_1 = wanted_nucleotide
            current_cycle = []
            while current_nucleotide_2 != wanted_nucleotide:
                current_cycle.append(dna_1.index(current_nucleotide_1))
                current_nucleotide_2 = dna_2[dna_1.index(current_nucleotide_1)]
                current_nucleotide_1 = dna_1[dna_1.index(current_nucleotide_2)]
            cycles.append(current_cycle)
    print(cycles)
    return combine_dna(dna_1, dna_2, cycles)
            


if __name__ == '__main__':
    tasks_data = upload_tasks()
    products = extract_products_instructions(tasks_data)
    # extract_resources_schedules(products)
    print(random.sample(range(0, 50), 50))
    dna_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    dna_2 = [9, 3, 7, 8, 2, 6, 5, 1, 4]
    print(crossover(dna_1, dna_2))
