import numpy as np
import pandas as pd

dataset2 = pd.read_csv("dataset2.csv", header="infer")
inputs = np.array(dataset2[["x1"]])
outputs = np.array(dataset2["y"])

def f(input):
    return input[0]**2

sq_errors = np.zeros(len(outputs))
for i in range(len(outputs)):
    error = f(inputs[i]) - outputs[i]
    sq_errors[i] = error**2
print(np.mean(sq_errors))