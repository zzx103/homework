import math
from collections import Counter


class Node:
    def __init__(self, label=None):
        self.label = label
        self.subnodes = {}
        self.result = {
            'label:': self.label,
            'snodes': self.subnodes
        }

    def add_node(self, f_val, node):
        self.subnodes[f_val] = node

    def predict(self, features, f_name):
        if len(self.subnodes) == 0:
            return self.label
        for i in range(len(f_name)):
            if f_name[i] == self.label:
                return self.subnodes[features[i]].predict(features, f_name)

    def __repr__(self):
        return '{}'.format(self.result)


def D_entropy(data, f_id):
    f_count = {}
    for k in data:
        if k[f_id] not in f_count:
            f_count[k[f_id]] = 1
        else:
            f_count[k[f_id]] += 1
    n = len(data)
    res = -sum([(i / n) * math.log2(i / n) for i in f_count.values()])
    return res


def condition_entropy(data, f_id):
    c_data = {}
    for k in data:
        if k[f_id] not in c_data:
            c_data[k[f_id]] = [k]
        else:
            c_data[k[f_id]].append(k)
    n = len(data)
    res = sum([(len(i) / n) * D_entropy(i, -1) for i in c_data.values()])
    return res


def max_g(data):
    ent = D_entropy(data, -1)
    g = []
    for i in range(len(data[0]) - 1):
        c_ent = condition_entropy(data, i)
        temp = ent - c_ent
        g.append((i, temp))
    res = max(g, key=lambda x: x[-1])
    return res


def Dtree_train(data, feature, epsilon=0.1):
    labels = [i[-1] for i in data]
    if len(set(labels)) == 1:
        return Node(label=labels[0])
    if len(feature) == 0:
        most_label = Counter(labels).most_common()[0][0]
        return Node(label=most_label)

    m_fid, m_g = max_g(data)
    m_feature = feature[m_fid]
    if m_g < epsilon:
        most_label = Counter(labels).most_common()[0][0]
        return Node(label=most_label)

    node_tree = Node(label=m_feature)

    s_feature = feature[:]
    s_feature.pop(m_fid)

    s_data = {}
    for k in data:
        temp = k[:]
        temp.pop(m_fid)
        if k[m_fid] not in s_data:
            s_data[k[m_fid]] = [temp]
        else:
            s_data[k[m_fid]].append(temp)

    for f, s_d in s_data.items():
        sub_tree = Dtree_train(s_d, s_feature)
        node_tree.add_node(f, sub_tree)

    return node_tree


datasets = [['青年', '否', '否', '一般', '否'],
            ['青年', '否', '否', '好', '否'],
            ['青年', '是', '否', '好', '是'],
            ['青年', '是', '是', '一般', '是'],
            ['青年', '否', '否', '一般', '否'],
            ['中年', '否', '否', '一般', '否'],
            ['中年', '否', '否', '好', '否'],
            ['中年', '是', '是', '好', '是'],
            ['中年', '否', '是', '非常好', '是'],
            ['中年', '否', '是', '非常好', '是'],
            ['老年', '否', '是', '非常好', '是'],
            ['老年', '否', '是', '好', '是'],
            ['老年', '是', '否', '好', '是'],
            ['老年', '是', '否', '非常好', '是'],
            ['老年', '否', '否', '一般', '否']]
features = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']

data_m = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']
    ]
features_m = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '类别']


m = Dtree_train(data_m, features_m)


