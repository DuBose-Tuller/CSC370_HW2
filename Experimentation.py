from Equation import *
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# TODO fix issue where it stops on the last eval

# Uncomment for dataset 1
dataset1 = pd.read_csv("dataset1.csv", header="infer")
inputs = np.array(dataset1["x"]).reshape(-1, 1) # dataset needs to be 2D
outputs = np.array(dataset1["f(x)"])
VARS = ["x"]

# Uncomment for dataset 2
# dataset2 = pd.read_csv("dataset2.csv", header="infer")
# inputs = np.array(dataset2[["x1", "x2", "x3"]])
# outputs = np.array(dataset2["y"])
# VARS = ["x", "y", "z"]

# # Scale stuff
# inputs[:, 0] = np.interp(inputs[:, 0], (inputs[:, 0].min(), inputs[:, 0].max()), (0, 100))
# inputs[:, 1] = np.interp(inputs[:, 1], (inputs[:, 1].min(), inputs[:, 1].max()), (0, 100))
# inputs[:, 2] = np.interp(inputs[:, 2], (inputs[:, 2].min(), inputs[:, 2].max()), (-1, +1))


new_eqn_params = {
    "one_operator": 0.3,
    "two_operators": 0.7,
    "val_is_x": 0.4,
    "min_val": -5,
    "max_val": 5,
    "max_depth": 2,
    "is_real": False,
    "start_depth": 0,
    "variables": VARS,
}

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.25)

def initialize(size, params):
    return [Equation(params) for i in range(size)]

NUM_GENERATIONS = 25
SIZE = 25
MUTATION_PROB = 0.3
CROSSOVER_PROB = 0.4
PARSIMONY = 8
NUM_CONTESTANTS = 2
RANDOM_INJECTION = 0.1
BROOD_SIZE = 1

current_gen = initialize(SIZE, new_eqn_params)
best_in_each_gen = []

for t in tqdm(range(NUM_GENERATIONS)):
    weights = []
    next_gen = []

    # Determine current gen's fitnesses
    for x in tqdm(current_gen):
        x.set_MSE(x_train, y_train, variables = VARS)
        reg_penalty = PARSIMONY * len(x.nodes)
        fitness = x.MSE + reg_penalty
        weights.append(1/fitness)

    # Create the next gen
    for j in range(len(current_gen)):
        parent1 = random.choices(current_gen, weights)[0]
        flip = random.random()
        if flip <= MUTATION_PROB:  # mutation
            next_gen.append(parent1.mutate(new_eqn_params))
        elif flip <= CROSSOVER_PROB + MUTATION_PROB:  # crossover
            parent2 = random.choices(current_gen, weights)[0]
            brood = parent1.crossover(parent2, brood=BROOD_SIZE)
            next_gen.extend(brood)
        else:  # clone
            next_gen.append(parent1)

    # Perform the selection tournament
    # TODO: it's currently based off of only MSE, maybe fix to include regularization
    selected = []
    for j in range(SIZE):
        if j < SIZE * RANDOM_INJECTION:
            new_rand_eqn = Equation(new_eqn_params)
            new_rand_eqn.set_MSE(x_train, y_train, variables = VARS)
            selected.append(new_rand_eqn)

        else:
            contestants = random.choices(next_gen, k=NUM_CONTESTANTS)
            selected.append(min(contestants, key=lambda t: t.MSE))

    current_gen = selected   

    # DEBUG: get this generation's best individual
    best_eqn = min(current_gen, key=lambda t: t.MSE)
    print(f"After Generation {t}, the best equation is...\n{best_eqn.root}\nIt has an MSE of {x.MSE}")
    

print(f"Final result: \n{best_eqn.root}\nMSE: {best_eqn.set_MSE(x_test, y_test, variables = VARS, set=False)}")