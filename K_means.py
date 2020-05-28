import math
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
                temp_dist = self.cal_dist(d, self.centers[i])
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    min_idx = i
            g[min_idx].append(d)
        return g

    def de_centers(self, groups_):
        new_centers = []
        for i in range(self.k):
            new_centers.append(self.cal_center(groups_[i]))
        return np.array(new_centers)

    def train(self, iter_times=100):
        if self.k >= len(self.data):
            return 'k>=n'
        for i_t in range(iter_times):
            groups = self.de_group()
            g_centers = self.de_centers(groups)
            if (g_centers == self.centers).all():  # 质心没更新, 提前结束
                break
            self.centers = g_centers
        return


def cluster_result(centers, data):
    res = []
    for d in data:
        min_dist = float('inf')
        min_idx = 0
        for i in range(centers.shape[0]):
            temp_dist = np.linalg.norm(d - centers[i])
            if temp_dist < min_dist:
                min_dist = temp_dist
                min_idx = i
        res.append(min_idx)
    return np.array(res)


def read_data_labels(s_labels, file):
    data = []
    label = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.strip().split(',')
            if temp[-1] in s_labels:
                dr = [float(i) for i in temp[:-1]]
                data.append(dr)
                label.append(s_labels.index(temp[-1]))
    return np.array(data), np.array(label)


def NMI(A,B):
    # len(A) should be equal to len(B)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #Mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps, 2)
    # Normalized Mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps, 2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat


if __name__ == '__main__':
    tr_file = 'avila-tr.txt'
    labels = ['B', 'F', 'I']
    tr_data, tr_label = read_data_labels(labels, tr_file)

    k = 3
    k_ms = k_means(tr_data, k)
    k_ms.train()
    print(k_ms.centers)
    cres = cluster_result(k_ms.centers, tr_data)
    print('NMI:', NMI(cres, tr_label))

