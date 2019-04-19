import numpy as np
from DataParser import DataParser

data_filenames = ["netflix-prize-data/combined_data_1.txt", "netflix-prize-data/combined_data_2.txt",
                  "netflix-prize-data/combined_data_3.txt", "netflix-prize-data/combined_data_4.txt"]

for index, filename in enumerate(data_filenames):
    data = DataParser(filename).parse()
    np.save("data_{}.npy".format(index), data)