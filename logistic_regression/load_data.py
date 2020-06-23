import numpy as np


class PreProcess:
    def __init__(self, data_path, split_str='\t'):
        self.data_path = data_path
        self.split_str = split_str

    def load_data(self):
        data, label = [], []
        f = open(self.data_path)
        lines = f.readlines()
        for line in lines:
            line_arr = line.strip().split(self.split_str)
            length = len(line_arr)
            label.append(line_arr[-1])
            # add 1 for bias
            data.append([1.] + line_arr[0:length - 1])
        return np.array(data, dtype=np.float32), np.array(label, dtype=np.float32)