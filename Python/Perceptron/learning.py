import perceptron
import data_importer
import sys
from random import randint
import matplotlib.pyplot as plt
import numpy as np

# AND GATE
# a b bias output
# 0 0 1    0
# 0 1 1    0
# 1 0 1    0
# 1 1 1    1

and_data = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
and_output = [0.0,0.0,0.0,1.0]

# OR GATE
# a b bias output
# 0 0 1    0
# 0 1 1    1
# 1 0 1    1
# 1 1 1    1

or_data = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
or_output = [0,1,1,1]


# NOT GATE
# a bias output
# 0 1    1
# 1 1    0

not_data = [[0,1],[1,1]]
not_output = [1,0] 

# sample 
# x1 x2 bias out
# 1.0 1.0 1 1 \
# 9.4 6.4 1 -1 \
# 2.5 2.1 1 1 \
# 8.0 7.7 1 -1 \
# 0.5 2.2 1 1 \
# 7.9 8.4 1 -1 \
# 7.0 7.0 1 -1 
# 2.8 0.8 1 1
# 1.2 3.0 1 1
# 7.8 6.1 1 -1
s_data = [[1.0, 1.0, 1.0], [9.4,6.4,1.0], [2.5,2.1,1.0], [8.0,7.7,1.0], [0.5,2.2,1.0], [7.9, 8.4, 1.0], [7.0, 7.0, 1.0] , [2.8,0.8,1.0],[1.2,3.0,1.0],[7.8,6.1,1.0]]
s_output = [1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,1.0,0.0]

plot_data_x = []
plot_data_y = []

for point in s_data:
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




# learning_graph.scatter(plot_data_x, plot_data_y)
# learning_graph.plot(line)
# plot_standard()
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
        print(str(result) + " = " + str(exp))

    accuracy = (correct / len(s_output)) * 100
    print("Accuracy: " + str(accuracy))

def learning(inputs, outputs, function, derivative_function):
    p = perceptron.Perceptron(function) 
    learning_rate = 1
    global_error = 20
    previous_weights = [p.get_weights()]
    previous_local_error = 20
    iteration = 0

    while global_error != 0.5:
        results = []
        i = 0
        wrong = 0

        for input in inputs:
            inp,o = generate_input_pair(inputs, outputs)
            result = p.run(inp)
            before_activation = p.sum(inp)
            results.append(result)
            local_error = o - result
            pW = ""
            # print(p.get_weights())
            if result != o:
                # deltaWi = a * ( y - h ) * xi
                # a = learning rate
                # y is the desired output
                # h is the actual output
                # xi is the input
                new_w = [y + (learning_rate * x * local_error * derivative_function(before_activation)) for x,y in zip(inp, p.get_weights())]
                # new_w = [round(y +round((learning_rate * x * local_error), 3), 3) for x,y in zip(inp, p.get_weights())]
                # print(new_w)
                pW = new_w
                p.setWeights(new_w)
                previous_local_error = local_error
                wrong += 1
            printRound(inp, result, o, pW)
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

        # global_error = overall_error

        error_list.append(overall_error)
        global_error_list.append(global_error)

        if len(error_list) > 20:
            error_list.pop(0)

        if len(global_error_list) > 20:
            global_error_list.pop(0)
        
        # weights = p.get_weights()
        # m = calculate_gradient(weights[1], weights[0], weights[2])
        # w, q = produce_points(m, weights[2])
        # line = np.linspace(w,q)

        learning_graph.clear()
        plot_check(p, inputs)
        # learning_graph.plot(line)
        # learning_graph.set_xlim([0,10])
        # learning_graph.set_ylim([0,10])

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




# t = learning(or_data,or_output, perceptron.simple)
# t = learning(not_data,not_output)

# p2 = perceptron.Perceptron(perceptron.sigmoid, [-1.13, -1.1, 10.9])


# print(str(p2.run([0,0,1])) + " should 0")
# print(str(p2.run([0,1,1])) + " should 1")
# print(str(p2.run([1,0,1])) + " should 1")
# print(str(p2.run([1,1,1])) + " should 1")

# print(str(p2.run([0,1])))

# print(str(p2.run([8.3,9.2,1])) + " should -1")


# # and
# t = learning(and_data,and_output, perceptron.sigmoid, perceptron.sigmoidDerivative)
# p2 = perceptron.Perceptron(perceptron.sigmoid, t)
# check_accuracy(p2, and_data, and_output)

# s_data
# t = learning(s_data,s_output, perceptron.sigmoid)
# p2 = perceptron.Perceptron(perceptron.sigmoid, t)
# check_accuracy(p2, s_data, s_output)

data = data_importer.read_file("./data/sample.csv")
t = learning(data.inputs,data.outputs, perceptron.sigmoid, perceptron.sigmoidDerivative)
p2 = perceptron.Perceptron(perceptron.sigmoid, t)
check_accuracy(p2, s_data, s_output)

print(p2.get_weights())




