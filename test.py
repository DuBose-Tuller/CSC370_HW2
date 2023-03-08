from Equation import *
import random
import pandas as pd
from tqdm import tqdm

new_eqn_params = {
    "one_operator": 0.3,
    "two_operators": 0.6,
    "val_is_x": 0.4,
    "min_val": -5,
    "max_val": 5,
    "max_depth": 1,
    "is_real": False,
    "start_depth": 0,
}


dataset1 = pd.read_csv("dataset1.csv", header="infer")
xs = list(dataset1["x"])
ys = list(dataset1["f(x)"])

def initialize(size, params):
    return [Equation(params) for i in range(size)]

NUM_GENERATIONS = 5
SIZE = 1000
MUTATION_PROB = 0.1
CROSSOVER_PROB = 0.6

current_gen = initialize(SIZE, new_eqn_params)
best_in_each_gen = []

for t in tqdm(range(NUM_GENERATIONS)):
    weights = []
    next_gen = []

    for x in current_gen:
        weights.append(1/x.get_fitness(xs, ys))

    for j in range(SIZE):
        parent1 = random.choices(current_gen, weights)[0]
        flip = random.random()
        if flip <= MUTATION_PROB:  # mutation
            next_gen.append(parent1.mutate(new_eqn_params))
        elif flip <= CROSSOVER_PROB + MUTATION_PROB:  # crossover
            parent2 = random.choices(current_gen, weights)[0]
            next_gen.extend(parent1.crossover(parent2))
        else:  # clone
            next_gen.append(parent1)

    current_gen = next_gen   

    best_fitness = max(weights)
    best_in_each_gen.append(best_fitness)
    


# Evaluate the MSE of each of the best equations in each gen
for eqn in best_in_each_gen:
    print(eqn)