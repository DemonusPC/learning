import math

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return round(1 / (1 + z),1)
    else:
        z = math.exp(x)
        return round(z / (1 + z),1)

def sigmoidDerivative(x):
    return sigmoid(1-sigmoid(x))

def simple(x):
    if x >= 0.5:
        return 1
    else:
        return 0

def x_simple(x):
    if x >= 5.0:
        return 1
    else:
        return -1