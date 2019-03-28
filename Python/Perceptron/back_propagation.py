import perceptron
import data_importer

from random import randint
import matplotlib.pyplot as plt
import numpy as np

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
        

# print("Reversed")
# for layer in reversed(neural_network):
#     print(layer)


# print(output_layer)
# sk = calc_sk(output_layer.get_nodes()[0].get_weights(), output_layer.get_inputs())
# delta_o = delta_output(error, perceptron.sigmoidDerivative, sk)


learning_rate = 2
# deltaj = (yj - hj) * g'(Σk[ wk,j * ak ])
def update_output_layer(layer, error):
    for node in layer.get_nodes():
        # print(layer.get_nodes().index(node))
        sk = calc_sk(node.get_weights(), layer.get_inputs())
        delta = delta_output(error, perceptron.sigmoidDerivative, sk)
        layer.add_delta(delta)
        update_weights(node, learning_rate,layer.inputs, delta)

# deltaj g'(Σk[ wk,j * ak ]) * Σi[ wj,i * deltai ]
def update_hidden_layer(layer):
    for node in layer.get_nodes():
        index = layer.get_nodes().index(node)
        sk = calc_sk(node.get_weights(), layer.get_inputs())

        # Get the neurons of the next node
        next_layer_nodes = layer.get_next().get_nodes()

        # Get the delta values for the next node
        next_deltas = layer.get_next().get_deltas()

        weights_linked_to_current = []

        for n in next_layer_nodes:
            v = n.get_weights()[index]
            weights_linked_to_current.append(v)

        delta = delta_hidden(perceptron.sigmoidDerivative, sk, weights_linked_to_current, next_deltas)
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
p1 = perceptron.Perceptron(perceptron.sigmoid, weights=[0.4,0.3,0.9], id=4)
p2 = perceptron.Perceptron(perceptron.sigmoid, weights=[0.8, -0.2, 0.5], id= 5)
# p1 = perceptron.Perceptron(perceptron.sigmoid, weights=[1,1,1], id=4)
# p2 = perceptron.Perceptron(perceptron.sigmoid, weights=[0.2, -0.111, 0.32], id= 5)
hidden_nodes = [p1,p2]
hidden_layer = Column(hidden_nodes)

# Create the output layer columns
output_layer = Column([perceptron.Perceptron(perceptron.sigmoid, weights=[-0.3, 0.1, 0.8], id=7)])

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

# for l in neural_network:
#     print(l)


# print("_______________________")
# print("Reversed")
for index, layer in enumerate(reversed(neural_network)):
    if(index == 0):
        update_output_layer(layer,error)
        print(layer)
    else:
        update_hidden_layer(layer)
        print(layer)


# input_functions
# Generate an input and output pairs 


# plot_data_x = []
# plot_data_y = []

# for point in d.inputs:
#     plot_data_x.append(point[0])
#     plot_data_y.append(point[1])

# fig = plt.figure(figsize=(7,10))
# # learning_graph = fig.add_subplot(211)
# # learning_graph.set_title("Data and learning")
# error_graph = fig.add_subplot(211)
# error_graph.set_title("Global Error")

# error_list = [20]
# global_error_list = [20]
# error_graph.plot(error_list)
# error_graph.plot(global_error_list)

# error_graph.set_ylim([-10,10])
# # learning_graph.axis([0.0, 10.0, 0.0, 10.0])


# plt.ion()
# plt.show()

# def pick_one_at_random(inputs, outputs):
#     index = randint(0, len(inputs)-1)
#     input = [inputs[index]]
#     output = [outputs[index]]
#     return input,output[0]



# for l in neural_network:
#     print(l)



# print("===========================")

# global_error = 100
# err = 1000.0

# epoch = 0
# while (err != 0.0 or epoch < 2000):
#     # input_layer.forward(pass_forward)
#     # print(float(err))
#     i, o = pick_one_at_random(d.inputs, d.outputs)
#     # for i, o in zip(d.inputs, d.outputs):
#     # print(i)
#     # print(o)
#     for layer in neural_network:
#         layer.forward(run_neuron)

#     # Get the results of the run
#     result = out.get_inputs()[0]
#     # Calculate the error
#     err = o - result

#     print("Expected: %s, Outpu: %s , Error: %s" % (str(o), str(result), str(err)))

#     # for l in neural_network:
#     #     print(l)


#     # print("_______________________")
#     # print("Reversed")
#     for index, layer in enumerate(reversed(neural_network)):
#         if(index == 0):
#             update_output_layer(layer,err)
#             layer.get_next().clear_cache()
#             # print(layer)
#         else:
#             update_hidden_layer(layer)
#             layer.get_next().clear_cache()
#             # print(layer)
#     input_layer.get_next().clear_cache()


#     if(err < global_error):
#         if round(learning_rate,1) > 0.2:
#             learning_rate -= 0.01
#             learning_rate = round(learning_rate, 3)
#         global_error = err

#     if err == 0.0:
#         epoch += 1

#     error_list.append(err) 
#     global_error_list.append(global_error)

#     if len(error_list) > 20:
#         error_list.pop(0)

#     if len(global_error_list) > 20:
#         global_error_list.pop(0)
#     error_graph.clear()
#     error_graph.plot(error_list)
#     error_graph.plot(global_error_list)
#     error_graph.set_ylim([-10,10])

#     plt.draw()
#     plt.pause(0.01)

# for l in neural_network:
#     print(l)

