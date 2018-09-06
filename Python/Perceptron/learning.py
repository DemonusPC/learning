import perceptron
# AND GATE
# a b bias output
# 0 0 1    0
# 0 1 1    0
# 1 0 1    0
# 1 1 1    1

# OR GATE
# a b bias output
# 0 0 1    0
# 0 1 1    1
# 1 0 1    1
# 1 1 1    1

# NOT GATE
# a bias output
# 0 1    1
# 1 1    0


and_data = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
and_output = [0,0,0,1]

or_data = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
or_output = [0,1,1,1]

not_data = [[0,1],[1,1]]
not_output = [1,0] 

epoch_target = 6

def learning(inputs, outputs):
    p = perceptron.Perceptron(perceptron.simple) 
    epoch = 0
    epoch_flag = False

    while(epoch_flag == False and epoch <= epoch_target):
        i = 0

        for input in inputs:
            result = p.run(input)
            printRound(input, result, outputs[i])
            if result != outputs[i]:
                epoch_flag = True
                if should_add(outputs[i], result):
                    new_w = [x+y for x,y in zip(input, p.get_weights())]
                    print(new_w)
                    p.setWeights(new_w)
                else:
                    new_w = [y-x for x,y in zip(input, p.get_weights())]
                    print(new_w)
                    p.setWeights(new_w)
            i += 1
        
        if epoch_flag == False:
            epoch += 1
            print("Epoch passed")
        elif epoch_flag == True:
            #epoch += 1
            epoch_flag = False
            #if epoch == 4:
            #   break

    print("Learning completed!")
    print("Weights" + str(p.get_weights()))
    return p.get_weights()

# should the value be added or subtracted
def should_add(expected, output):
    if(expected >= output):

        return True
    else:
        return False

def printRound(input, output, expected):
    stt = "wrong"
    if(output == expected):
        stt = "correct"
    print(str(input) + ": " + str(output) + " - " + str(expected) + " -> " + stt)

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


