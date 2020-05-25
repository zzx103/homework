import math
from collections import namedtuple


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
        k = len(data[0])
        dm_new = (dm + 1) % k
        data.sort(key=lambda x: x[dm])
        im = len(data) // 2
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
    n_point = node.value
    #     如果目标节点分割维度的值小于当前节点
    #     目标点离左子树更近
    if t_point[n_dem] < n_point[n_dem]:
        nearer_node = node.left
        further_node = node.right
    #     目标点离右子树更近
    else:
        nearer_node = node.right
        further_node = node.left

    #     递归找到目标点所在的区域
    travel(nearer_node, t_point, nst, k)

    #     计算欧式距离
    dist = cal_dist(t_point, n_point)

    # 如果L的长度小于k，直接添加
    if len(nst['nearest']) < k:
        nst['nearest'].append(n_point)
        nst['dist'].append(dist)

    else:
        # 如果当前结点与目标结点的距离大于L的最大距离，跳出
        if dist > max(nst['dist']):
            return
        # 如果当前结点与目标结点的距离小于L的最大距离，替换
        if len(nst['nearest']) == k and dist < nst['max_dist']:
            idx = nst['dist'].index(nst['max_dist'])
            nst['nearest'][idx] = n_point
            nst['dist'][idx] = dist

    # 检查另外一个节点是否有更近的点
    travel(further_node, t_point, nst, k)
    return


def find_k_nearest(tree, target, k):
    nearest = {'nearest': [], 'dist': []}
    travel(tree.root, target, nearest, k)
    return nearest


if __name__ == "__main__":
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    kdtree = kd_tree(data)
    K = 3
    res = find_k_nearest(kdtree, [3, 4.5], K)
    print(res)
