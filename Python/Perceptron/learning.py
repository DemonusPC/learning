import perceptron
import sys
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
# x1 x2 out
# 1.0 1.0 1 \
# 9.4 6.4 -1 \
# 2.5 2.1 1 \
# 8.0 7.7 -1 \
# 0.5 2.2 1 \
# 7.9 8.4 -1 \
# 7.0 7.0 -1 
# 2.8 0.8 1
# 1.2 3.0 1
# 7.8 6.1 -1
s_data = [[1.0, 1.0, 1.0], [9.4,6.4,1.0], [2.5,2.1,1.0], [8.0,7.7,1.0], [0.5,2.2,1.0], [7.9, 8.4, 1.0], [7.0, 7.0, 1.0] , [2.8,0.8,1.0],[1.2,3.0,1.0],[7.8,6.1,1.0]]
s_output = [1,-1,1,-1,1,-1,1,-1,-1,1,1,-1]



epoch_target = 5


def learning(inputs, outputs):
    p = perceptron.Perceptron(perceptron.simple) 
    learning_rate = 1
    err = 100

    while(err != 0):
        i = 0
        results = []

        for input in inputs:
            result = p.run(input)
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
                new_w = [round(y +round((learning_rate * x * error), 2), 2) for x,y in zip(input, p.get_weights())]
                # print(new_w)
                pW = new_w
                p.setWeights(new_w)
            printRound(input, result, outputs[i], pW)
            i += 1


        err = calculate_error(outputs, results)
        
        print(err)
        if(err == 0): 
            break

        

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

t = learning(and_data,and_output)
# t = learning(or_data,or_output)
# t = learning(not_data,not_output)
p2 = perceptron.Perceptron(perceptron.simple, t)

print(str(p2.run([0,0,1])) + " should 0")
print(str(p2.run([0,1,1])) + " should 0")
print(str(p2.run([1,0,1])) + " should 0")
print(str(p2.run([1,1,1])) + " should 1")

# print(str(p2.run([0,0,1])) + " should 0")
# print(str(p2.run([0,1,1])) + " should 1")
# print(str(p2.run([1,0,1])) + " should 1")
# print(str(p2.run([1,1,1])) + " should 1")

# print(str(p2.run([0,1])))


