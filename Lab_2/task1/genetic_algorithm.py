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
            products[int(column/2)].append([tasks_data.iloc[row, column], tasks_data.iloc[row, column + 1]])
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
        
    

if __name__ == '__main__':
    tasks_data = upload_tasks()
    products = extract_products_instructions(tasks_data)
    extract_resources_schedules(products)
    
