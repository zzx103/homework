
import numpy as np





class kmeans():

    @staticmethod
    def cal_center(Gi):
        return np.mean(Gi, axis=0)

    @staticmethod
    def cal_dist(x, y, p=2):
        return np.linalg.norm(x-y, ord=p)

    def train(self, train_data, k, iter_times=100):
        if k >= len(train_data):
            return
        groups = []
        g_center = []
        for i in range(k):
            groups.append([])
            g_center.append(train_data[i])

        for i_t in range(iter_times):
            for d in train_data:
                min_dist = float('inf')
                min_idx = 0
                for i in range(len(g_center)):
                    dist = self.cal_dist(g_center[i], d)
                    if dist < min_dist:
                        min_idx = i
                groups[min_idx].append(d)

            temp_center = []
            for i in range(k):
                temp_center.append(self.cal_center(groups[i]))
            dif = self.cal_dist(temp_center, g_center)
            if dif == 0 or i_t == iter_times - 1:
                return groups
            else:
                g_center = temp_center
                groups.clear()
