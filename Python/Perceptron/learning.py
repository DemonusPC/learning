import perceptron
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
and_output = [0,0,0,1]

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
s_output = [1,-1,1,-1,1,-1,1,-1,-1,1,1,-1]

plot_data_x = []
plot_data_y = []

for point in s_data:
    plot_data_x.append(point[0])
    plot_data_y.append(point[1])

line = np.linspace(9.5,0,10)
line = np.linspace(-1.13,)

plt.scatter(plot_data_x, plot_data_y)
plt.plot(line)
plt.axis([0.0, 10.0, 0.0, 10.0])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()



epoch_target = 5


def learning(inputs, outputs):
    p = perceptron.Perceptron(perceptron.x_simple) 
    learning_rate = 0.2
    err = 100
    err_flag = False
    epoch = 0

    while err != 0 or epoch < epoch_target:
        i = 0
        results = []

        for input in inputs:
            index = randint(0, len(inputs)-1)
            result = p.run(inputs[index])
            results.append(result)
            error = round(outputs[i] - result, 2)
            pW = ""
            # print(p.get_weights())
            if result != outputs[i]:
                # deltaWi = a * ( y - h ) * xi
                # a = learning rate
                # y is the desired output
                # h is the actual output
                # xi is the input
                new_w = [round(y +round((learning_rate * x * error), 2), 2) for x,y in zip(inputs[index], p.get_weights())]
                # print(new_w)
                pW = new_w
                p.setWeights(new_w)
                err_flag = True
            printRound(input, result, outputs[i], pW)
            i += 1


        err = calculate_error(outputs, results)
        if err_flag == False:
            epoch += 1
            learning_rate -= 0.05
            print("raising epoch")
        else:
            print("fail")
            err_flag = False
            epoch = 0
            learning_rate = 0.2
        
        print(err)
        print(epoch)
        print(epoch_target)
        if epoch < epoch_target:
            print("Should not continue")
        
        if epoch < epoch_target or err != 0:
            print("Should not continue")
        

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

t = learning(s_data,s_output)
# t = learning(or_data,or_output)
# t = learning(not_data,not_output)
p2 = perceptron.Perceptron(perceptron.x_simple, t)

# print(str(p2.run([0,0,1])) + " should 0")
# print(str(p2.run([0,1,1])) + " should 0")
# print(str(p2.run([1,0,1])) + " should 0")
# print(str(p2.run([1,1,1])) + " should 1")

# print(str(p2.run([0,0,1])) + " should 0")
# print(str(p2.run([0,1,1])) + " should 1")
# print(str(p2.run([1,0,1])) + " should 1")
# print(str(p2.run([1,1,1])) + " should 1")

# print(str(p2.run([0,1])))

print(str(p2.run([8.3,9.2,1])) + " should -1")



