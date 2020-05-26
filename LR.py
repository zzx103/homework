import math
import numpy as np


class LR:
    def __init__(self, tr_data, tr_label):
        self.x = self.data_matrix(tr_data)
        self.y = tr_label
        self.d_num, self.f_num = self.x.shape
        self.weight = np.zeros((len(self.x[0]), 1), dtype=np.float)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def data_matrix(self, data):
        temp_one = [1.0] * len(data)
        re_data = np.insert(data, 0, values=temp_one, axis=1)
        return re_data

    def train(self, iter_times=100, learning_rate=0.01):
        # label = np.mat(y)

        for t_iter in range(iter_times):
            for i in range(self.d_num):
                error = self.sigmoid(np.dot(self.x[i], self.weight)) - self.y[i]
                self.weight += -learning_rate * error * np.transpose([self.x[i]])

    # def f(self, x):
    #     return -(self.weights[0] + self.weights[1] * x) / self.weights[2]

    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weight)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)


if __name__ == "__main__":
    test = [[1, 2, 3], [4, 5, 6]]
    ntest = np.array(test, dtype=np.float)
    print(ntest)

