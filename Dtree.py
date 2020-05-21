import math
from collections import Counter
from random import shuffle


# 计算信息熵
def D_entropy(data, f_id):
    # 统计同一属性的不同值出现次数
    f_count = {}
    for k in data:
        if k[f_id] not in f_count:
            f_count[k[f_id]] = 1
        else:
            f_count[k[f_id]] += 1
    # 计算信息熵
    n = len(data)
    res = -sum([(i / n) * math.log2(i / n) for i in f_count.values()])
    return res


# 计算条件信息熵
def condition_entropy(data, f_id):
    # 将不同属性值的数据进行分组
    c_data = {}
    for k in data:
        if k[f_id] not in c_data:
            c_data[k[f_id]] = [k]
        else:
            c_data[k[f_id]].append(k)
    # 计算条件信息熵
    n = len(data)
    res = sum([(len(i) / n) * D_entropy(i, -1) for i in c_data.values()])
    return res


# 计算最大信息增益
def max_g(data):
    # 计算数据集的信息熵
    ent = D_entropy(data, -1)
    g = []
    # 计算不同属性的信息增益
    for i in range(len(data[0]) - 1):
        c_ent = condition_entropy(data, i)
        temp = ent - c_ent
        g.append((i, temp))
    # 求最大信息增益
    res = max(g, key=lambda x: x[1])
    return res

# 树节点
class Node:
    def __init__(self, label=None):
        # 节点标签
        self.label = label
        # 节点子树
        self.subnodes = {}
        self.result = {
            '标签：': self.label,
            '子树：': self.subnodes
        }

    def add_node(self, f_val, node):
        # 将值为f_val的子树添加进来
        self.subnodes[f_val] = node

    def classify(self, features, f_name):
        # 节点无子树，返回节点自身标签
        if len(self.subnodes) == 0:
            return self.label
        # 找到对应属性值的子树，由子树进行预测
        for i in range(len(f_name)):
            if f_name[i] == self.label:
                if features[i] in self.subnodes:
                    return self.subnodes[features[i]].classify(features, f_name)
        # 找不到对应属性值，无法进行决策
        return 'unable'

    def __repr__(self):
        return '{}'.format(self.result)


# ID3算法生成决策树
def Dtree_train(data, feature, epsilon=0.1):
    label_value = [i[-1] for i in data]
    # 情况1：所有标签只有一个值
    if len(set(label_value)) == 1:
        return Node(label=label_value[0])
    # 情况2：属性集为空
    if len(feature) == 0:
        most_label = Counter(label_value).most_common()[0][0]
        return Node(label=most_label)
    # 情况3
    # 计算最大信息增益
    m_fid, m_g = max_g(data)
    m_feature = feature[m_fid]
    # 信息增益小于阈值
    if m_g < epsilon:
        most_label = Counter(label_value).most_common()[0][0]
        return Node(label=most_label)
    # 建立新节点，标签为信息增益最大属性
    node_tree = Node(label=m_feature)
    # 从属性集中去除信息增益最大属性
    s_feature = feature[:]
    s_feature.pop(m_fid)
    # 将数据按信息增益最大属性的值进行分组
    s_data = {}
    for k in data:
        temp = k[:]
        temp.pop(m_fid)
        if k[m_fid] not in s_data:
            s_data[k[m_fid]] = [temp]
        else:
            s_data[k[m_fid]].append(temp)
    # 用子数据集和子属性集递归生成子树
    for f, s_d in s_data.items():
        sub_tree = Dtree_train(s_d, s_feature)
        node_tree.add_node(f, sub_tree)

    return node_tree


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

# k折交叉验证
# 先打乱数据集
shuffle(data_m)
k_fold = 3
# 将数据集平均分为k份，每次选一份作为验证集，其余为训练集
# 数据集无法均分余下的数据加入训练集
for i in range(k_fold):
    ibeg = len(data_m) // k_fold * i
    iend = ibeg + len(data_m) // k_fold
    v_data = data_m[ibeg:iend]
    t_data = data_m[:ibeg] + data_m[iend:]
    m = Dtree_train(t_data, features_m)
    s = len(v_data)
    hc = 0
    uc = 0
    for vd in v_data:
        result = m.classify(vd, features_m)
        if result == vd[-1]:
            hc += 1
        # 数据集较小，训练好的决策树可能缺少某些属性值，对具有该属性值的数据无法进行决策
        elif result == 'unable':
            uc += 1
    print(m)
    print('验证集准确率：', hc/s)

