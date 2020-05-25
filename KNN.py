import math
from collections import Counter


class kd_node():
    def __init__(self, dm, value, left, right):
        self.dem = dm
        self.value = value
        self.left = left
        self.right = right


class kd_tree():
    def __init__(self, data):
        self.root = self.build(data, 0)

    def build(self, data, dm):
        if len(data) == 0:
            return None
        k = len(data[0]) - 1
        data.sort(key=lambda x: x[dm])
        im = len(data) // 2
        dm_new = (dm + 1) % k
        left = self.build(data[:im], dm_new)
        right = self.build(data[im + 1:], dm_new)
        return kd_node(dm, data[im], left, right)


# KDTree的前序遍历
def preorder(root):
    print(root.value)
    if root.left:  # 节点不为空
        preorder(root.left)
    if root.right:
        preorder(root.right)


def cal_dist(x, y, p=2):
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1 / p)
    else:
        return 0


def travel(node, t_point, nst, k):
    if node is None:
        return

    #     当前节点的分割维度
    n_dem = node.dem
    #     当前节点的值
    n_point = node.value[:-1]
    #     如果目标节点分割维度的值小于当前节点
    #     目标点离左子树更近
    if t_point[n_dem] < n_point[n_dem]:
        nearer_node = node.left
        further_node = node.right
    #     目标点离右子树更近
    else:
        nearer_node = node.right
        further_node = node.left

    # 递归找到目标点所在的区域
    travel(nearer_node, t_point, nst, k)

    # 计算欧式距离
    dist = cal_dist(t_point, n_point)

    # 如果L的长度小于k，直接添加
    if len(nst['nearest']) < k:
        nst['nearest'].append(node.value)
        nst['dist'].append(dist)

    else:
        # 如果当前结点与目标结点的距离大于L的最大距离，跳出
        if dist > max(nst['dist']):
            return
        # 如果当前结点与目标结点的距离小于L的最大距离，替换
        if dist < max(nst['dist']):
            max_idx, _ = max(enumerate(nst['dist']), key=lambda x: x[1])
            nst['nearest'][max_idx] = node.value
            nst['dist'][max_idx] = dist

    # 检查另外一个节点是否有更近的点
    travel(further_node, t_point, nst, k)
    return


def find_k_nearest(tree, target, k):
    nearest = {'nearest': [], 'dist': []}
    travel(tree.root, target, nearest, k)
    return nearest


if __name__ == "__main__":
    data = [[2, 3, 'a'], [5, 4, 'a'], [9, 6, 'a'], [4, 7, 'b'], [8, 1, 'b'], [7, 2, 'b']]
    kdtree = kd_tree(data)
    K = 3
    res = find_k_nearest(kdtree, [4, 5], K)
    print(res)
    label_value = [d[-1] for d in res['nearest']]
    most_label = Counter(label_value).most_common()[0][0]
    print(most_label)

