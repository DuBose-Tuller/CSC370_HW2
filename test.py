from Equation import *


params = {
    "one_operator": 0.3,
    "two_operators": 0.6,
    "val_is_x": 0.4,
    "min_val": -5,
    "max_val": 5,
    "max_depth": 1,
    "is_real": False,
    "start_depth": 0,
}

def initialize(size, params):
    return [Equation(params) for i in range(size)]

first_gen = initialize(2, params)
next_gen = first_gen[0].crossover(first_gen[1])

print("")