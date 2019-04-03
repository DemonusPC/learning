import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# sns.set()
# tips = pd.read_csv('./data/xor.csv')
# print(tips)

# print(type(tips))

# ax = sns.scatterplot(x="a", y="b", hue="out", style="out", data=tips)


# plt.show()

class Reporter: 
  def __init__(self, data_path, initial_error= 100):
    sns.set()

    self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 8))

    data = pd.read_csv(data_path)
    

    err_d = [[0, initial_error]]
    self.error_data = pd.DataFrame(err_d, columns=['epoch', 'error'])
    # print(self.error_data)
    self.error_graph = sns.lineplot(x="epoch", y="error", data=self.error_data, ax=self.axes[0])

    # self.errAni = animation.FuncAnimation(self.fig, self.plot_error, interval=500)

    # data graph
    self.data_graph = sns.scatterplot(x="a", y="b", hue="out", style="out", data=data, ax=self.axes[1])

    # self.error_graph.set(ylim=(-2, 20))
    self.data_graph.set(ylim=(-1, 2), xlim=(-1,2))


    # machine graph
    # represents the result set after running the neural network
    self.machine_data = data
    self.machine_graph = sns.scatterplot(x="a", y="b", hue="out", style="out", data=self.machine_data, ax=self.axes[2])
    self.machine_graph.set(ylim=(-1, 2), xlim=(-1,2))
    # print(data)
    # print(self.machine_data)
  
  def run(self):
    plt.ion()
    plt.show()

  def add_machine_set(self, inputs, output):
    if len(self.machine_data) >= 20:
      self.machine_data = self.machine_data.iloc[:10]

    pp = inputs + output
    data = pd.DataFrame([pp], columns = ['a', 'b', 'out'])
    new_data = self.machine_data.append(data)
    self.machine_data = new_data
    self.machine_graph.clear()
    self.machine_graph = sns.scatterplot(x="a", y="b", hue="out", data=self.machine_data, ax=self.axes[2], legend=False)
    # self.machine_graph = sns.pairplot(self.machine_data)
    self.machine_graph.set(ylim=(-1, 2), xlim=(-1,2))

  def add_error(self, epoch, error):
    # If our error graph is above 20 elements remove one element from the front
    if len(self.error_data) >= 20:
      self.error_data = self.error_data.iloc[1:]
    df2 = pd.DataFrame([[epoch, error]], columns=['epoch', 'error'])
    err = self.error_data.append(df2)
    self.error_data = err
    self.plot_error()

  def plot_error(self):
    self.error_graph.clear()
    self.error_graph = sns.lineplot(x="epoch", y="error", data=self.error_data, ax=self.axes[0])
    plt.draw()
    plt.pause(0.001)

class Logger:
  @staticmethod
  def log_run(expected, output, error):
    print("Expected: %s, Output: %s , Error: %s" % (str(expected), str(output), str(error)))

  @staticmethod
  def log_test_set(expected, result, correct):
    if(expected == result):
      print(str(result) + " = " + str(expected) + " - " + u'\u2713')
    else: 
      print(str(result) + " = " + str(expected) + " - x")

  @staticmethod
  def log_success(expected, result):
    print(str(result) + " = " + str(expected) + " - " + u'\u2713')

  @staticmethod
  def log_failure(expected, result):
    print(str(result) + " = " + str(expected) + " - x")

  @staticmethod
  def log_accuracy(accuracy):
    print('Accuracy: ' + str(accuracy))
