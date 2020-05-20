import numpy as np


def train(x, y, w, b, r):
    while True:
        # 误分类个数
        miscount = 0

        for i in range(len(x)):
            if -1 * y[i] * (np.dot(w, x[i]) + b) > 0:
                # 更新权重和偏差
                w = w + r * y[i] * x[i]
                b = b + r * y[i]

                miscount = miscount + 1

        if miscount == 0:
            break

    return w, b


# 学习率
learning_rate = 0.1

# and函数训练样本
x_and = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])

y_and = np.array([1, -1, -1, -1])
# 随机初始化参数
weight_and = np.random.random((1, 2))
bias_and = np.random.random()

weight_and, bias_and = train(x_and, y_and, weight_and, bias_and, learning_rate)

print('and函数:')
for i in range(len(x_and)):
    print(x_and[i], end=':')
    if np.dot(weight_and, x_and[i]) + bias_and > 0:
        print('true')
    else:
        print('false')

# or函数训练样本
x_or = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y_or = np.array([1, 1, 1, -1])
# 随机初始化参数
weight_or = np.random.random((1, 2))
bias_or = np.random.random()

weight_or, bias_and = train(x_or, y_or, weight_or, bias_or, learning_rate)

print('or函数:')
for i in range(len(x_or)):
    print(x_or[i], end=':')
    if np.dot(weight_or, x_or[i]) + bias_and > 0:
        print('true')
    else:
        print('false')



