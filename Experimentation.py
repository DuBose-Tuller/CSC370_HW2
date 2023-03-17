from Equation import *
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


VARS = ["x", "y", "z"]

new_eqn_params = {
    "one_operator": 0.3,
    "two_operators": 0.6,
    "val_is_x": 0.6,
    "min_val": -5,
    "max_val": 5,
    "max_depth": 5,
    "is_real": True,
    "start_depth": 0,
    "variables": VARS,
}


dataset2 = pd.read_csv("dataset2.csv", header="infer")
inputs = np.array(dataset2[["x1", "x2", "x3"]])

# Scale x1 and x2 down
inputs[:, 0] = np.interp(inputs[:, 0], (inputs[:, 0].min(), inputs[:, 0].max()), (-1, +1))
inputs[:, 1] = np.interp(inputs[:, 1], (inputs[:, 1].min(), inputs[:, 1].max()), (-1, +1))
inputs[:, 2] = np.interp(inputs[:, 2], (inputs[:, 2].min(), inputs[:, 2].max()), (-1, +1))

outputs = np.array(dataset2["y"])
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.25)

def initialize(size, params):
    return [Equation(params) for i in range(size)]

NUM_GENERATIONS = 5
SIZE = 100
MUTATION_PROB = 0.2
CROSSOVER_PROB = 0.5
PARSIMONY = 4
NUM_CONTESTANTS = 4
RANDOM_INJECTION = 0.1

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
    for j in range(SIZE):
        parent1 = random.choices(current_gen, weights)[0]
        flip = random.random()
        if flip <= MUTATION_PROB:  # mutation
            next_gen.append(parent1.mutate(new_eqn_params))
        elif flip <= CROSSOVER_PROB + MUTATION_PROB:  # crossover
            parent2 = random.choices(current_gen, weights)[0]
            next_gen.extend(parent1.crossover(parent2))
            # next_gen.append(parent1.crossover(parent2))
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

    best_eqn = min(current_gen, key=lambda t: t.MSE)


    print(f"After Generation {t}, the best equation is...\n{best_eqn.root}\nIt has an MSE of {x.MSE}")
    

print(f"Final result: \n{best_eqn.root}\nMSE: {best_eqn.set_MSE(x_test, y_test, variables = VARS, set=False)}")