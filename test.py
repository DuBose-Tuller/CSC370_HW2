from Equation import *


params = {
    "one_operator": 0.3,
    "two_operators": 0.6,
    "val_is_x": 0.25,
    "min_val": -5,
    "max_val": 5,
    "max_depth": 3,
    "is_real": False
}

def initialize(size, is_real, max_depth, threshold):
    return [Equation(params) for i in range(size)]

eqn = Equation(params)
eqn2 = eqn.copy()

eqn2.left.value = 5
print("")