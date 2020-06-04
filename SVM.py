import numpy as np


class SVM:
    def __init__(self, train_data, labels):
        self.d_num, self.f_num = train_data.shape
        self.x = train_data
        self.y = labels
        self.b = 0.0
        self.alpha = np.ones(self.d_num)
        self.E = [self._E(i) for i in range(self.d_num)]
        self.C = 1.0

    # 核函数
    def _K(self, xi, xj):
        return np.dot(xi, xj)

    # 计算E(i)
    def _E(self, i):
        return self._g(i) - self.y[i]

    # 计算g(i)
    def _g(self, i):
        res = self.b
        for j in range(self.d_num):
            res += self.alpha[j] * self.y[j] * self._K(self.x[j], self.x[i])
        return res

    # 判断是否满足KKT条件
    def _KKT(self, i):
        y_g = self._g(i) * self.y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    # 选择α1和α2
    def _choose_alpha(self):
        i_a1 = -1
        i_a2 = -1
        for i in range(self.d_num):
            if 0 < self.alpha[i] < self.C:
                if not self._KKT(i):
                    i_a1 = i
                    break
        if i_a1 == -1:
            for i in range(self.d_num):
                if not self._KKT(i):
                    i_a1 = i
                    break
        if i_a1 == -1:
            return i_a1, i_a2
        if self.E[i_a1] > 0:
            i_a2 = min(range(self.d_num), key=lambda x: self.E[x])
        else:
            i_a2 = max(range(self.d_num), key=lambda x: self.E[x])
        return i_a1, i_a2

    # 开始训练
    def train(self, i_times=100):
        for t in range(i_times):
            i_a1, i_a2 = self._choose_alpha()
            if i_a1 == -1:
                break
            a1 = self.alpha[i_a1]
            a2 = self.alpha[i_a2]
            x1 = self.x[i_a1]
            x2 = self.x[i_a2]
            y1 = self.y[i_a1]
            y2 = self.y[i_a2]
            E1 = self.E[i_a1]
            E2 = self.E[i_a2]

            if y1 != y2:
                L = max(0, a2 - a1)
                H = min(self.C, self.C + a2 - a1)
            else:
                L = max(0, a2 + a1 - self.C)
                H = min(self.C, a2 + a1)

            eta = self._K(x1, x1) + self._K(x2, x2) - 2 * self._K(x1, x2)
            a2_new_unc = a2 + y2 * (E1 - E2) / eta
            if a2_new_unc > H:
                a2_new = H
            elif a2_new_unc < L:
                a2_new = L
            else:
                a2_new = a2_new_unc
            a1_new = a1 + y1 * y2 * (a2 - a2_new)
            b1_new = -E1 - y1 * self._K(x1, x1) * (a1_new - a1) - y2 * self._K(x2, x1) * (a2_new - a2) + self.b
            b2_new = -E2 - y1 * self._K(x1, x2) * (a1_new - a1) - y2 * self._K(x2, x2) * (a2_new - a2) + self.b

            if 0 < a1_new < self.C:
                b_new = b1_new
            elif 0 < a2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i_a1] = a1_new
            self.alpha[i_a2] = a2_new
            self.b = b_new

            self.E[i_a1] = self._E(i_a1)
            self.E[i_a2] = self._E(i_a2)

    # 判断是正例还是负例
    def classify(self, data):
        r = self.b
        for i in range(self.d_num):
            r += self.alpha[i] * self.y[i] * self._K(data, self.x[i])
        if r > 0:
            return 1
        else:
            return -1

    # 计算测试集上的准确率
    def score(self, x_test, y_test):
        right_count = 0
        for i in range(len(x_test)):
            result = self.classify(x_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(x_test)

    # 获取权重
    def weight(self):
        yx = self.y.reshape(-1, 1) * self.x
        w = np.dot(yx.T, self.alpha)
        return w


# 读取数据
def read_two_label(la, lb, file):
    data = []
    label = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.strip().split(',')
            if temp[-1] == la or temp[-1] == lb:
                dr = [float(i) for i in temp[:-1]]
                data.append(dr)
                label.append(1 if temp[-1] == la else -1)
    return np.array(data), np.array(label)


if __name__ == '__main__':
    tr_file = 'avila-tr.txt'
    ts_file = 'avila-ts.txt'

    label_1 = 'E'
    label_2 = 'F'

    tr_data, tr_label = read_two_label(label_1, label_2, tr_file)
    ts_data, ts_label = read_two_label(label_1, label_2, ts_file)

    my_svm = SVM(tr_data, tr_label)
    my_svm.train(i_times=500)
    res = my_svm.score(ts_data, ts_label)
    print('precision: ', res)

