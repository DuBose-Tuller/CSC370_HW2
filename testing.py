from Equation import *

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

eqn = Equation(new_eqn_params)

print(eqn.root)