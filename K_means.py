
import numpy as np


class k_means():
    def __init__(self, train_data, k):
        self.data = np.array(train_data)
        self.k = k
        self.centers = self.data[:k]

    @staticmethod
    def cal_center(gi):
        return np.mean(np.array(gi), axis=0)

    @staticmethod
    def cal_dist(x, y, p=2):
        return np.linalg.norm(x-y, ord=p)

    def de_group(self):
        g = []
        for i in range(self.k):
            g.append([])
        for d in self.data:
            min_dist = float('inf')
            min_idx = 0
            for i in range(self.k):
                if self.cal_dist(d, self.centers[i]) < min_dist:
                    min_idx = i
            g[min_idx].append(d)
        return g

    def de_centers(self, groups_):
        new_centers = []
        for i in range(self.k):
            new_centers.append(self.cal_center(groups_[i]))
        return np.array(new_centers)

    def train(self, train_data, k, iter_times=100):
        if k >= len(train_data):
            return 'k>=n'
        for i_t in range(iter_times):
            groups = self.de_group()
            g_centers = self.de_centers(groups)
            if (g_centers == self.centers).all():  # 质心没更新, 提前结束
                break
            self.centers = g_centers
        return


def read_two_label(s_labels, file):
    data = []
    label = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.strip().split(',')
            if temp[-1] in s_labels:
                dr = [float(i) for i in temp[:-1]]
                data.append(dr)
                label.append()
    return np.array(data), np.array(label)


if __name__ == '__main__':
    tr_file = 'avila-tr.txt'

    label_1 = 'E'
    label_2 = 'F'

    tr_data, tr_label = read_two_label(label_1, label_2, tr_file)
    k_ms = k_means(tr_data, 2)
    print(k_ms.centers)