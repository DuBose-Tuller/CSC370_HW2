from Equation import *
import pandas as pd

VARS = ["x", "y", "z"]

new_eqn_params = {
    "one_operator": 0,
    "two_operators": 1,
    "val_is_x": 0.65,
    "min_val": 0,
    "max_val": 100,
    "max_depth": 1,
    "is_real": False,
    "start_depth": 0,
    "variables": VARS,
}

eqn = Equation(new_eqn_params)

print(eqn.root)

dataset2 = pd.read_csv("dataset2.csv", header="infer")
inputs = np.array(dataset2[["x1", "x2", "x3"]])
outputs = np.array(dataset2["y"])


print(eqn.set_MSE(inputs, outputs, variables=VARS, set=False))