from Equation import *
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


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
# inputs[:, 2] = np.interp(inputs[:, 2], (inputs[:, 2].min(), inputs[:, 2].max()), (0, 100))


new_eqn_params = {
    "one_operator": 0.1,
    "two_operators": 0.3,
    "val_is_x": 0.5,
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

NUM_GENERATIONS = 20
SIZE = 200
MUTATION_PROB = 0.2
MUTATION_PROB_REGROW = 0.2
CROSSOVER_PROB = 0.5
PARSIMONY = 0.2 # multiplies the MSE by this time time the number of nodes as a reg penaly. 0 == off
NUM_CONTESTANTS = 2 # 1 turns it off
RANDOM_INJECTION = 0.0
SEMANTIC_THRESHOLD = -1 # 0 prevents exact duplicates
SEMANTIC_PROP = 0.01

current_gen = initialize(SIZE, new_eqn_params)
best_in_each_gen = []


def calculate_percent_diff(parent, candidate, prop=1):
    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    indices = indices[:int(len(y_train)*prop)]
    sample_in = x_train[indices]

    parent_vals = np.array([parent.evaluate(x, VARS, parent.root) for x in sample_in])
    candidate_vals = np.array([candidate.evaluate(x, VARS, candidate.root) for x in sample_in])

    parent_vals[parent_vals == 0] = 1 # safe division TODO improve
    mean_diff = np.mean(np.abs((parent_vals-candidate_vals))/np.abs(parent_vals))
    return mean_diff

for t in tqdm(range(NUM_GENERATIONS)):
    weights = []
    next_gen = []

    # Determine current gen's fitnesses
    for x in tqdm(current_gen):
        x.set_MSE(x_train, y_train, variables = VARS)
        penalty = x.MSE + (x.MSE*PARSIMONY*len(x.nodes))
        weights.append(1/(1+penalty))

    # Create the next gen
    next_gen.append(min(current_gen, key=lambda t: t.MSE)) # auto include the single best
    for x in tqdm(current_gen):
        parent1 = random.choices(current_gen, weights)[0]
        #parent1 = x
        flip = random.random()
        if flip <= MUTATION_PROB:  # mutation
            next_gen.append(parent1.mutate(new_eqn_params, MUTATION_PROB_REGROW))
        elif flip <= CROSSOVER_PROB + MUTATION_PROB:  # crossover
            parent2 = random.choices(current_gen, weights)[0]

            # Ensure that the second parent isn't too similar to the first
            semantics = calculate_percent_diff(parent1, parent2, prop=SEMANTIC_PROP)
            while semantics < np.log10(parent2.MSE)/SEMANTIC_THRESHOLD: # More stringent for larger MSE values
            # while semantics < SEMANTIC_THRESHOLD:
                print("Diversity Fail!")
                parent2 = random.choices(current_gen, weights)[0]
                semantics = calculate_percent_diff(parent1, parent2, prop=SEMANTIC_PROP)

            next_gen.extend(parent1.crossover(parent2))
            #next_gen.append(parent1.crossover(parent2)[0])
        else:  # clone
            next_gen.append(parent1)


    # Perform the selection tournament
    # TODO: it's currently based off of only MSE, maybe fix to include regularization
    # TODO: prevent some of the same equations from being overly resampled
    selected = []

    for j in tqdm(range(SIZE)):
        if j < SIZE * RANDOM_INJECTION:
            new_rand_eqn = Equation(new_eqn_params)
            new_rand_eqn.set_MSE(x_train, y_train, variables = VARS)
            selected.append(new_rand_eqn)

        else:
            contestants = random.choices(next_gen, k=NUM_CONTESTANTS)
            selected.append(min(contestants, key=lambda t: t.MSE))

    current_gen = selected   

    # Uncomment for no tournament
    # current_gen = next_gen

    # DEBUG: get this generation's best individual
    best_eqn = min(current_gen, key=lambda t: t.MSE)
    print(f"After Generation {t}, the best equation is...\n{best_eqn.root}\nIt has an MSE of {x.MSE} which has {int(np.log10(x.MSE+0.001)//1)+1} digits.")
    

print(f"Final result: \n{best_eqn.root}\nMSE: {best_eqn.set_MSE(x_test, y_test, variables = VARS, set=False)}")