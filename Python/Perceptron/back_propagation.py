import perceptron
import data_importer
from functions import p_sigmoid, p_sigmoidDerivative
from random import randint
# import matplotlib.pyplot as plt
# import numpy as np

from reporter import Reporter

def read_in_data():
    data = data_importer.read_file("./data/xor.csv")
    return data

def calc_sk(weights, inputs):
    result = 0
    for w, a in zip(weights,inputs):
        result = result + (w*a)
    return result

def delta_output(error, derivative, sk):
    
    delta = error * derivative(sk)
    return delta

def delta_hidden(derivative, sk, next_weights, next_deltas ):
    result = derivative(sk)
    next_sum = [w * d for w,d in zip(next_weights, next_weights)]
    
    s = 0
    for ss in next_sum:
        s += ss
    result = result * s
    return result

def add_bias(input_list):
    input_list.append(1)
    return input_list

def pass_forward(node,input):
    return node

def run_neuron(node, input):
    if(isinstance(node,int)):
        return node
    result = node.run(input)
    return result

# wi,j = wi,j + α ai Δj
def update_weights(neuron, alpha, inputs, delta):
    new_w = [weight + (alpha * input * delta) for weight,input in zip(neuron.get_weights(), inputs)]
    neuron.set_weights(new_w)

class Column:
    # self , nodes, inputs, next
    def __init__(self, nodes):
        self.nodes = nodes
        self.inputs = []
        self.next = None
        self.deltas = []
    
    def __str__(self):
        result = "--- Column --- \n"
        result += str(self.inputs)
        result += "\n"
        result += str(self.nodes)
        result += "\n"
        result += str(self.deltas)
        return result

    def __repr__(self):
        return "Column"
        
    # methods
    
    # Sets the inuts from the previous column
    def set_inputs(self, inputs):
        self.inputs = inputs

    # Sets the next column to follow
    def set_next(self, next):
        self.next = next

    # Returns the neurons that belong to this column
    def get_nodes(self):
        return self.nodes

    def get_inputs(self):
        return self.inputs

    def get_next(self):
        return self.next

    def forward(self, function):
        result = []
        for node in self.nodes:
            r = function(node, self.inputs)
            result.append(r)
        add_bias(result)
        self.next.set_inputs(inputs=result)
    
    def set_deltas(self, new_deltas):
        self.deltas = new_deltas

    def add_delta(self, new_delta):
        self.deltas.append(new_delta)

    def get_deltas(self):
        return self.deltas

    def clear_cache(self):
        self.deltas = []
        self.inputs = []

# deltaj = (yj - hj) * g'(Σk[ wk,j * ak ])
def update_output_layer(layer, error):
    for node in layer.get_nodes():
        # print(layer.get_nodes().index(node))
        sk = calc_sk(node.get_weights(), layer.get_inputs())
        delta = delta_output(error, p_sigmoidDerivative, sk)
        layer.add_delta(delta)
        update_weights(node, learning_rate,layer.inputs, delta)

# deltaj g'(Σk[ wk,j * ak ]) * Σi[ wj,i * deltai ]
def update_hidden_layer(layer):
    for node in layer.get_nodes():
        index = layer.get_nodes().index(node)
        print("index: " + str(index))
        sk = calc_sk(node.get_weights(), layer.get_inputs())

        # Get the neurons of the next node
        next_layer_nodes = layer.get_next().get_nodes()

        # Get the delta values for the next node
        next_deltas = layer.get_next().get_deltas()

        weights_linked_to_current = []

        for n in next_layer_nodes:
            v = n.get_weights()[index]
            weights_linked_to_current.append(v)

        delta = delta_hidden(p_sigmoidDerivative, sk, weights_linked_to_current, next_deltas)
        layer.add_delta(delta)
        update_weights(node, learning_rate, layer.inputs, delta)



print("Read in the xor data")
d = read_in_data()

# Create the input sets
input_set = d.inputs[0]
output_set = d.outputs[0]

# Create the input layer coluimn
input_layer = Column(input_set)

# Create the hidden layer columns
p1 = perceptron.Perceptron(p_sigmoid, weights=[0.4,0.3,0.9], id=4)
p2 = perceptron.Perceptron(p_sigmoid, weights=[0.8, -0.2, 0.5], id= 5)
# p1 = perceptron.Perceptron(perceptron.sigmoid, weights=[1,1,1], id=4)
# p2 = perceptron.Perceptron(perceptron.sigmoid, weights=[0.2, -0.111, 0.32], id= 5)
hidden_nodes = [p1,p2]
hidden_layer = Column(hidden_nodes)

# Create the output layer columns
output_layer = Column([perceptron.Perceptron(p_sigmoid, weights=[-0.3, 0.1, 0.8], id=7)])

# Create the output (not layer)
out = Column([])

# Add the links between the pages
input_layer.set_next(hidden_layer)
hidden_layer.set_next(output_layer)
output_layer.set_next(out)

# Set up the input layer, pass forward the values
input_layer.forward(pass_forward)

neural_network = [hidden_layer, output_layer]

# Go forward in the neural network
for layer in neural_network:
    layer.forward(run_neuron)

# Get the results of the run
result = out.get_inputs()[0]

# Calculate the error
error = output_set - result

print("Expected: %s, Outpu: %s , Error: %s" % (str(output_set), str(result), str(error)))


# input_functions
# Generate an input and output pairs 
def pick_one_at_random(inputs, outputs):
    index = randint(0, len(inputs)-1)
    input = [inputs[index]]
    output = [outputs[index]]
    return input,output[0]



for l in neural_network:
    print(l)



print("===========================")
learning_rate = 2
global_error = 100
err = 1000.0

report = Reporter('./data/xor.csv')
report.run()

epoch = 0
while (err != 0.0 and epoch < 2000):
    print(epoch)
    # input_layer.forward(pass_forward)
    i, o = pick_one_at_random(d.inputs, d.outputs)
    for layer in neural_network:
        layer.forward(run_neuron)

    # Get the results of the run
    result = out.get_inputs()[0]
    # Calculate the error
    err = o - result

    print("Expected: %s, Output: %s , Error: %s" % (str(o), str(result), str(err)))

    # for l in neural_network:
    #     print(l)


    # print("_______________________")
    # print("Reversed")
    for index, layer in enumerate(reversed(neural_network)):
        if(index == 0):
            update_output_layer(layer,err)
            # layer.get_next().clear_cache()
            # print(layer)
        else:
            update_hidden_layer(layer)
            # layer.get_next().clear_cache()
            print(layer)
    # input_layer.get_next().clear_cache()


    if(abs(err) < abs(global_error)):
        if round(learning_rate,10) > 0.2:
            learning_rate -= 0.2
            learning_rate = round(learning_rate, 10)
        global_error = err

    print(learning_rate)
    print(global_error)

    # clean the cache
    for layer in neural_network:
        layer.get_next().clear_cache()
    input_layer.get_next().clear_cache()

    epoch += 1

    report.add_error(epoch, err)
    

for l in neural_network:
    print(l)

