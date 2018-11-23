import perceptron
import data_importer

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
    result = result * next_sum
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
        

d = read_in_data()

input_set = d.inputs[0]
output_set = d.outputs[0]
# print(input_set)
# print(output_set)

p1 = perceptron.Perceptron(perceptron.sigmoid, weights=[0.4,0.3,0.9], id=4)
p2 = perceptron.Perceptron(perceptron.sigmoid, weights=[0.8, -0.2, 0.5], id= 5)
hidden_nodes = [p1,p2]

input_layer = Column(input_set)
hidden_layer = Column(hidden_nodes)
# print(input_layer)
# print(hidden_layer)
output_layer = Column([perceptron.Perceptron(perceptron.sigmoid, weights=[-0.3, 0.1, 0.8], id=7)])
out = Column([])

input_layer.set_next(hidden_layer)
hidden_layer.set_next(output_layer)
output_layer.set_next(out)

input_layer.forward(pass_forward)

# print(hidden_layer.get_inputs())

neural_network = [hidden_layer, output_layer]

for layer in neural_network:
    # print(layer)
    layer.forward(run_neuron)


result = out.get_inputs()[0]
error = output_set - result

print("Output: " + str(result) + " -> Error: " + str(error))


# print("Reversed")
# for layer in reversed(neural_network):
#     print(layer)


# print(output_layer)
# sk = calc_sk(output_layer.get_nodes()[0].get_weights(), output_layer.get_inputs())
# delta_o = delta_output(error, perceptron.sigmoidDerivative, sk)



def update_output_layer(layer):
    for node in layer.get_nodes():
        sk = calc_sk(node.get_weights(), layer.get_inputs())
        delta = delta_output(error, perceptron.sigmoidDerivative, sk)
        layer.add_delta(delta)
        update_weights(node, 1,layer.inputs, delta)

def update_hidden_layer(layer):
    for node in layer.get_nodes():
        sk = calc_sk(node.get_weights(), layer.get_inputs())
        delta = 

print("Reversed")
for index, layer in enumerate(reversed(neural_network)):
    if(index == 0):
        update_output_layer(layer)
        print(layer)
    else:
        print("hidden layer")

