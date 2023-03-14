from Equation import *
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


# 1. Simplify mutations
# 2. Improved Selection algorithm
# 3. Inject a small percentage of random trees each gen

new_eqn_params = {
    "one_operator": 0.3,
    "two_operators": 0.6,
    "val_is_x": 0.4,
    "min_val": -5,
    "max_val": 5,
    "max_depth": 2,
    "is_real": False,
    "start_depth": 0,
    "variables": "x",
}


dataset1 = pd.read_csv("dataset1.csv", header="infer")
xs = np.array(dataset1["x"])
ys = np.array(dataset1["f(x)"])
x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size = 0.25)

def initialize(size, params):
    return [Equation(params) for i in range(size)]

NUM_GENERATIONS = 2
SIZE = 20
MUTATION_PROB = 0.1
CROSSOVER_PROB = 0.6
PARSIMONY = 10

current_gen = initialize(SIZE, new_eqn_params)
best_in_each_gen = []

for t in tqdm(range(NUM_GENERATIONS)):
    weights = []
    next_gen = []

    for x in current_gen:
        x.set_MSE(x_train, y_train)
        reg_penalty = PARSIMONY * len(x.nodes)
        fitness = x.MSE + reg_penalty
        weights.append(1/fitness)

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

    best_eqn = min(current_gen, key=lambda t: t.MSE)


    print(f"After Generation {t}, the best equation is...\n{best_eqn.root}\nIt has an MSE of {x.MSE}")
    

print(f"Final result: \n{best_eqn.root}\nMSE: {best_eqn.set_MSE(x_test, y_test, set=False)}")