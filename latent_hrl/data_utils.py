
import csv
import matplotlib.pyplot as plt
import numpy as np

def load_data(input_filepath):
    data = []
    with open(input_filepath, 'rb') as infile:
        # remove header
        infile.readline()
        # read each line 
        csv_reader = csv.reader(infile)
        for row in csv_reader:
            data.append(int(row[-1]))
    data = np.asarray(data, dtype=np.int64)
    return data

def load_action_data(input_filepath):
    max_data = []
    cur_data = []
    with open(input_filepath, 'rb') as infile:
        for row in infile:
            row = row.strip()
            if row == '':
                if len(cur_data) > len(max_data):
                    max_data = cur_data
                cur_data = []
            else:
                cur_data.append(int(row))
    return np.array(max_data)

def plot_data(data):
    plt.plot(range(len(data)), data)
    plt.show()

if __name__ == '__main__':
    # input_filepath = '../data/old_faithful.csv'
    # data = load_data(input_filepath)
    # plot_data(data)
    # print data
    input_filepath = '/Users/wulfebw/Desktop/agent_0_actions.txt'
    load_action_data(input_filepath)