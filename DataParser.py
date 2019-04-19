import numpy as np

class DataParser():
    def __init__(self, filename):
        self.filename = filename

    def parse(self):
        index = 0
        with open(self.filename, "r") as file:
            self.data = file.read().strip().split("\n")
            self.total = len([0 for row in self.data if ":" not in row])
            self.result = np.zeros((self.total, 6))
            for row in self.data:
                if ":" in row:
                    movie_id = int(row.strip()[:-1])
                else:
                    user_id, rating, date = row.split(",")
                    user_id = int(user_id)
                    year, month, day = [int(x) for x in date.split("-")]
                    self.result[index] = [movie_id, user_id, rating, year, month, day]
                    index += 1
                    if index % 100000 == 0 or index == self.total - 1:
                        print("\r{} parsed {} %".format(self.filename, round(100 * index / self.total, 2)), end="\t")
            print("")
        return self.result