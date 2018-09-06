import math

class Perceptron:
    def __init__(self, activation, weights =[0,0,0]):
        self.weights = weights 
        self.activation = activation

    def get_weights(self):
        return self.weights
    
    def printList(self):
        st = "( "
        for x in self.weights:
            st = st + str(x) + " "

        st += ")"
    
    def setWeights(self, new_weights):
        self.weights = new_weights

    def run(self, input):
        s = self.sum(input)
        h = self.activation(s)
        return h

    def sum(self, input):
        s = 0
        #print(s)
        for i, w in zip(input, self.weights):
            # print (str(i) + " * " + str(w))
            s = s + (i * w)
        return s 

def acti(x, y):
    return x*y

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def simple(x):
    if x >= 0.5:
        return 1
    else:
        return 0
    

w = [0,0,0]
inp = [0,0,1]

p1 = Perceptron(simple)
p1.setWeights(w)
r = p1.run(inp)
print(r)




