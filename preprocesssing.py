"""
unzip netflix-prize-data.zip to the local working directory of this project
this script creates a numpy array data.npy for collaborative_filtering.py and saves it to hard disk

you should have at least 8 GB RAM in order to preprocess all of the netflix prize data
"""

import os
import numpy as np
from DataParser import DataParser

data_filenames = ["netflix-prize-data/combined_data_1.txt", "netflix-prize-data/combined_data_2.txt",
                  "netflix-prize-data/combined_data_3.txt", "netflix-prize-data/combined_data_4.txt"]

num_entries = []

for index, filename in enumerate(data_filenames):
    data = DataParser(filename).parse()
    np.save("data_{}.npy".format(index + 1), data)
    num_entries.append(len(data))
    del data

data = np.zeros((np.sum(num_entries), 6))
for i in range(4):
    print("processing numpy array {} ... ".format(i+1))
    low = int(np.sum(num_entries[:i]))
    high = int(np.sum(num_entries[:i+1]))
    data[low : high] = np.load("data_{}.npy".format(i+1))

np.save("data.npy", data)

for i in range(1, 5):
    os.remove("data_{}.npy".format(i))