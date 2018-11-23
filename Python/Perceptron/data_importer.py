import csv



class Data:
    def __init__(self, inputs, outputs, labels):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
    
    def get_label_index(self, label):
        return 0

# Adds 1 to the input array
def add_bias(input_list):
    input_list.append(1)
    return input_list

def process_first_row(reader):
    headers = next(reader, None)
    return headers

def parse_row_to_float(row):
    result = []
    for col in row:
        element = float(col)
        result.append(element)
    return result



def read_file(path):
    print("Starting file inport at path: " + path)
    with open(path, newline='') as csvfile:
     data_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
     
     print("Getting labels")
     labels = process_first_row(data_reader)
     inputs = []
     outputs = []

     print("Processing rows")
     for row in data_reader:
         output = float(row.pop())
         input_row = parse_row_to_float(row)
         # add_bias(input_row)
         inputs.append(input_row)
         outputs.append(output)
     result = Data(inputs, outputs, labels)
    
    print("Done!")
    return result