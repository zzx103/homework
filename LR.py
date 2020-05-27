import math
import numpy as np


class LR:
    def __init__(self, tr_data, tr_label):
        self.x = tr_data
        self.y = tr_label
        self.weight = np.zeros(tr_data.shape[1], dtype=np.float)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def train(self, iter_times=100, learning_rate=0.01):
        for i_t in range(iter_times):
            for xi, yi in zip(self.x, self.y):
                gradient = (self.sigmoid(np.dot(xi, self.weight)) - yi) * xi
                self.weight += -learning_rate * gradient

    def score(self, x_test, y_test, p=0.5):
        right_count = 0
        for x, y in zip(x_test, y_test):
            P_x1 = self.sigmoid(np.dot(x, self.weight))
            if (P_x1 >= p and y == 1) or (P_x1 < p and y == 0):
                right_count += 1
        return right_count / len(x_test)


def data_preprocess(data):
    temp_one = [1.0] * len(data)
    re_data = np.insert(data, 0, values=temp_one, axis=1)
    return re_data


if __name__ == "__main__":
    test = [[1, 2, 3], [4, 5, 6]]
    ntest = np.array(test, dtype=np.float)
    print(ntest)

