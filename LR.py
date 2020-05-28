import math
import numpy as np


class LR:
    def __init__(self, train_data, train_label):
        self.x = train_data
        self.y = train_label
        self.weight = np.zeros(train_data.shape[1], dtype=np.float)

    # sigmoid函数
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # 开始训练
    def train(self, iter_times=100, learning_rate=0.01):
        for i_t in range(iter_times):
            for xi, yi in zip(self.x, self.y):
                # 计算梯度
                gradient = (self.sigmoid(np.dot(xi, self.weight)) - yi) * xi
                # 更新权重
                self.weight += -learning_rate * gradient

    # 判断数据标签
    # 若P(x=1)大于p则返回1，否则返回0
    def classify(self, data, p=0.5):
        P_x1 = self.sigmoid(np.dot(data, self.weight))
        if P_x1 >= p:
            return 1
        else:
            return 0

    # 计算测试集上的准确率
    def score(self, x_test, y_test, pr=0.5):
        right_count = 0
        for x, y in zip(x_test, y_test):
            result = self.classify(x, p=pr)
            if result == y:
                right_count += 1
        return right_count / len(x_test)


# 数据预处理
# 在所有样本的特征向量前增加1
def data_preprocess(data):
    temp_one = [1.0] * len(data)
    re_data = np.insert(data, 0, values=temp_one, axis=1)
    return re_data


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
                label.append(1 if temp[-1] == la else 0)
    return np.array(data), np.array(label)


if __name__ == '__main__':
    tr_file = 'avila-tr.txt'
    ts_file = 'avila-ts.txt'

    label_1 = 'E'
    label_2 = 'F'

    tr_data, tr_label = read_two_label(label_1, label_2, tr_file)
    ts_data, ts_label = read_two_label(label_1, label_2, ts_file)

    tr_data = data_preprocess(tr_data)
    ts_data = data_preprocess(ts_data)

    lrc = LR(tr_data, tr_label)
    lrc.train()
    res = lrc.score(ts_data, ts_label)
    print(res)
