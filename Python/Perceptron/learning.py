import perceptron
import data_importer
import sys
from random import randint
import matplotlib.pyplot as plt
import numpy as np


data = data_importer.read_file("./data/sample.csv")

plot_data_x = []
plot_data_y = []

for point in data.inputs:
    plot_data_x.append(point[0])
    plot_data_y.append(point[1])

fig = plt.figure(figsize=(7,10))
learning_graph = fig.add_subplot(211)
learning_graph.set_title("Data and learning")
error_graph = fig.add_subplot(212)
error_graph.set_title("Global Error")



error_list = [20]
global_error_list = [20]
error_graph.plot(error_list)
error_graph.plot(global_error_list)


line = np.linspace(9.5,0,10)

def plot_standard():
    learning_graph.scatter(plot_data_x, plot_data_y)
    learning_graph.plot(line)

def createPlotLines(data):
    x = []
    y = []
    for point in data:
        x.append(point[0])
        y.append(point[1])
    return x,y

def plot_check(perceptron, inputs):
    clas0 = []
    clas1 = []
    for i in inputs:
        result = perceptron.run(i)
        if result == 0:
            clas0.append(i)
        else:
            clas1.append(i)

    c0x, c0y = createPlotLines(clas0)
    c1x, c1y = createPlotLines(clas1)

    learning_graph.scatter(c0x, c0y, color='red')
    learning_graph.scatter(c1x, c1y, color='blue')


def final_check(perceptron, inputs):
    clas0 = []
    clas1 = []
    for i in inputs:
        result = perceptron.run(i)
        if result == 0:
            clas0.append(i)
        else:
            clas1.append(i)

    c0x, c0y = createPlotLines(clas0)
    c1x, c1y = createPlotLines(clas1)

    plt.scatter(c0x, c0y, color='red')
    plt.scatter(c1x, c1y, color='blue')
    plt.show()

error_graph.set_ylim([0,20])
learning_graph.axis([0.0, 10.0, 0.0, 10.0])

plt.ion()
plt.show()

epoch_target = 5

def check_accuracy(perceptron, data, expected_output):
    correct = 0
    for inp, exp in zip(data, expected_output):
        result = perceptron.run(inp)
        if(result == exp):
            correct += 1
            print(str(result) + " = " + str(exp) + " - " + u'\u2713')
        else:
            print(str(result) + " = " + str(exp) + " - x")
    
    accuracy = (correct / len(data)) * 100
    print("Accuracy: " + str(accuracy))


# input_functions
# Generate an input and output pairs 
def pick_one_at_random(inputs, outputs):
    index = randint(0, len(inputs)-1)
    input = [inputs[index]]
    output = [outputs[index]]
    return input,output

def pass_training_set(inputs, outputs):
    return inputs, outputs

def pass_extended_ts(inputs, outputs):
    inp = inputs + inputs + inputs
    out = outputs + outputs + outputs
    return inp, out

def most_of_ts(inputs, outputs):
    input_result = inputs[:5]
    output_result = outputs[:5]
    return input_result, output_result



# learning
# this is a single perceptron learning routine

def learning(inputs, outputs, function, derivative_function, input_function):
    p = perceptron.Perceptron(function) 
    learning_rate = 1
    global_error = 20
    iteration = 0

    while global_error != 0:
        results = []
        i = 0
        wrong = 0
        
        # Get curated inputs
        c_input, c_output = input_function(inputs, outputs)

        # loop through the inputs and run the 
        for input, output in zip(c_input,c_output):
            result = p.run(input)
            before_activation = p.sum(input)
            results.append(result)
            local_error = output - result
            pW = ""
            if result != output:
                # deltaWi = a * ( y - h ) * xi
                # a = learning rate
                # y is the desired output
                # h is the actual output
                # xi is the input
                new_w = [y + (learning_rate * x * local_error * derivative_function(before_activation)) for x,y in zip(input, p.get_weights())]
                pW = new_w
                p.setWeights(new_w)
                wrong += 1
            printRound(input, result, output, pW)
            i += 1

        accuracy = ((i-wrong)/i)*100
        
        print("Accuracy: " + str(accuracy))
        overall_error = calculate_error(outputs, results)

        if(overall_error < global_error):
            if round(learning_rate,1) > 0.2:
                learning_rate -= 0.1
                learning_rate = round(learning_rate, 3)
            global_error = overall_error


        print("learning rate: " + str(learning_rate))

        error_list.append(overall_error)
        global_error_list.append(global_error)

        if len(error_list) > 20:
            error_list.pop(0)

        if len(global_error_list) > 20:
            global_error_list.pop(0)

        learning_graph.clear()
        plot_check(p, inputs)

        error_graph.clear()
        error_graph.plot(error_list)
        error_graph.plot(global_error_list)
        error_graph.set_ylim([0,20])

        plt.draw()
        plt.pause(0.01)

        iteration += 1

        print("global error: " + str(global_error))
        print(iteration)

        

    print("Learning completed!")
    print("Weights" + str(p.get_weights()))
    return p.get_weights()

def printRound(input, output, expected, weights):
    stt = "wrong"
    if(output == expected):
        stt = "correct"
    sys.stdout.write("\r {0}: {1} - {2}  ->  {3} - {4}\n".format(str(input), str(output), str(expected), stt, str(weights)))

def calculate_error(desired, output):
    # E= 0.5 Σi(yi –hi )2
    error = 0.5 * sum([(x-y)**2 for x, y in zip(desired, output)])
    return error

def generate_input_pair(inputs, outputs):
    index = randint(0, len(inputs)-1)
    input = inputs[index]
    output = outputs[index]
    return input,output


# linear calculation
# y, x, c
def calculate_gradient(x2, x1, bias):
    if x1 == 0:
        return x2-bias
    m = (x2-bias) / x1
    return m

def produce_points(gradient, bias):
    # y = mx+c
    first = bias
    second = (10 * gradient) + bias
    
    return first,second

t = learning(data.inputs,data.outputs, perceptron.sigmoid, perceptron.sigmoidDerivative, most_of_ts)
p2 = perceptron.Perceptron(perceptron.sigmoid, t)
check_accuracy(p2, data.inputs, data.outputs)

print(p2.get_weights())
