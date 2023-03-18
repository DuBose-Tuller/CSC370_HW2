import numpy as np
import pandas as pd

dataset2 = pd.read_csv("dataset2.csv", header="infer")
inputs = np.array(dataset2[["x1", "x2", "x3"]])
outputs = np.array(dataset2["y"])

def test(input):
    return (52*input[1])/(input[0])


print(np.mean([(test(inputs[i])-outputs[i])**2 for i in range(len(outputs))]))