import math
    

class Perceptron:
    def __init__(self, activation, weights =[0,0,0], id = 0):
        self.weights = weights 
        self.activation = activation
        self.id = id

    def __str__(self):
        return str(self.weights)
    
    def __repr__(self):
        return str(self.id) + " -> |" + str(self.weights) + "|"

    def get_id(self):
        return self.id

    def get_weights(self):
        return self.weights
    
    def printList(self):
        st = "( "
        for x in self.weights:
            st = st + str(x) + " "

        st += ")"
    
    def setWeights(self, new_weights):
        self.weights = new_weights
    
    def set_weights(self, new_weights):
        self.weights = new_weights

    def run(self, input):
        s = self.sum(input)
        h = self.activation(s)
        return h

    def sum(self, input):
        s = 0
        for i, w in zip(input, self.weights):
            s = s + (i * w)
        return s 

def acti(x, y):
    return x*y

