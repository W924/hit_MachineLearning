# coding:utf-8
import numpy as np
import random
import matplotlib.pyplot as plt


# 随机生成数据
def generate_data(opt):
    num = 100                      # num: 生成数据数量
    p = 0.5                        # p: Y=1数据所占的比例
    x_data = []
    y_data = []
    # opt = 0，生成的数据满足贝叶斯假设，且方差与Y独立
    if opt == 0:
        for i in range(int(num * p)):
            x1y1 = random.gauss(4, 1)
            x2y1 = random.gauss(4, 1)
            x_data.append([1.0, x1y1, x2y1])
            y_data.append(1)
        for i in range(int(num - num * p)):
            x1y0 = random.gauss(6, 1)
            x2y0 = random.gauss(7, 1)
            x_data.append([1.0, x1y0, x2y0])
            y_data.append(0)
    # opt = 1，生成的数据不满足贝叶斯假设，但是方差与Y的类别无关
    if opt == 1:
        mean1 = [4, 4]
        cov1 = [[1, 0.5], [0.5, 1]]
        mean2 = [6, 7]
        cov2 = [[1, 0.5], [0.5, 1]]
        x1y1, x2y1 = np.random.multivariate_normal(mean1, cov1, int(num * p)).T
        x1y0, x2y0 = np.random.multivariate_normal(mean2, cov2, int(num - num*p)).T
        for i in range(int(num * p)):
            x_data.append([1.0, x1y1[i], x2y1[i]])
            y_data.append(1)
        for i in range(int(num - num * p)):
            x_data.append([1.0, x1y0[i], x2y0[i]])
            y_data.append(0)
    # opt = 2，满足贝叶斯假设，但是方差与Y的类别相关
    if opt == 2:
        mean1 = [4, 4]
        cov1 = [[1, 0], [0, 1]]
        mean2 = [6, 7]
        cov2 = [[2, 0], [0, 2]]
        x1y1, x2y1 = np.random.multivariate_normal(mean1, cov1, int(num * p)).T
        x1y0, x2y0 = np.random.multivariate_normal(mean2, cov2, int(num - num * p)).T
        for i in range(int(num * p)):
            x_data.append([1.0, x1y1[i], x2y1[i]])
            y_data.append(1)
        for i in range(int(num - num * p)):
            x_data.append([1.0, x1y0[i], x2y0[i]])
            y_data.append(0)
    # opt = 3，不满足贝叶斯假设，且方差与Y的类别相关
    if opt == 3:
        mean1 = [4, 4]
        cov1 = [[1, 0.5], [0.5, 1]]
        mean2 = [6, 7]
        cov2 = [[2, 0.5], [0.5, 2]]
        x1y1, x2y1 = np.random.multivariate_normal(mean1, cov1, int(num * p)).T
        x1y0, x2y0 = np.random.multivariate_normal(mean2, cov2, int(num - num * p)).T
        for i in range(int(num * p)):
            x_data.append([1.0, x1y1[i], x2y1[i]])
            y_data.append(1)
        for i in range(int(num - num * p)):
            x_data.append([1.0, x1y0[i], x2y0[i]])
            y_data.append(0)
    return x_data, y_data


# 从文件中读取数据
def load_data():
    x_set = []
    y_set = []
    fp = open('data1.txt')
    for line in fp.readlines():
        line_array = line.strip().split()
        x_set.append([1.0, float(line_array[0]), float(line_array[1])])
        y_set.append(int(line_array[2]))
    return x_set, y_set


# 梯度上升法（不带正则项）
def grad_ascent_without_regular(x_train, y_train):
    alpha = 1e-2                       # 步长
    seita = 1e-5                       # 终止判断标准
    n = len(x_train[0])                 # 特征的维数 + 1
    count = 0                           # 迭代轮数
    weight_matrix = np.zeros((n, 1))    # 初始化参数
    weight_pre = np.zeros((n, 1))
    x_train_matrix = np.mat(x_train)
    y_train_matrix = np.mat(y_train).T
    while True:
        count = count + 1
        tmp = np.exp(x_train_matrix * weight_matrix) / (1 + np.exp(x_train_matrix * weight_matrix))
        distance = alpha * x_train_matrix.transpose() * (y_train_matrix - tmp)
        weight_matrix = weight_matrix + distance
        mcle_now = 0
        mcle_pre = 0
        # 计算沿梯度上升前后目标函数L(W)的值
        for i in range(len(y_train)):
            tmp = (weight_matrix.T * x_train_matrix[i].T)[0][0]
            mcle_now = mcle_now * i / (i + 1) + (y_train[i] * tmp - np.log(1 + np.exp(tmp))) / (i + 1)
            tmp = (weight_pre.T * x_train_matrix[i].T)[0][0]
            mcle_pre = mcle_pre * i / (i + 1) + (y_train[i] * tmp - np.log(1 + np.exp(tmp))) / (i + 1)
        # 如果两次的目标函数值L(W)之差非常小，就退出
        if np.abs(mcle_now - mcle_pre) < seita:
            break
        else:
            print "L(W) = " + str(mcle_now)
            weight_pre = weight_matrix
        if count > 1000:
            alpha = 1e-3
    return weight_matrix.transpose().tolist()[0], count


# 梯度上升（加正则项）
def grad_ascent_with_regular(x_train, y_train):
    lamda = 1e-2                # 惩罚项的系数
    alpha = 1e-2                # 步长
    seita = 1e-5                # 终止判断标准
    n = len(x_train[0])         # 特征的维数 + 1
    count = 0                   # 迭代轮数
    weight_matrix = np.zeros((n, 1))  # 初始化参数
    weight_pre = np.zeros((n, 1))
    x_train_matrix = np.mat(x_train)
    y_train_matrix = np.mat(y_train).T
    while True:
        count = count + 1
        tmp = np.exp(x_train_matrix * weight_matrix) / (1 + np.exp(x_train_matrix * weight_matrix))
        distance = alpha * x_train_matrix.transpose() * (y_train_matrix - tmp)
        weight_matrix = weight_matrix + distance - alpha * lamda * weight_matrix
        mcle_now = 0
        mcle_pre = 0
        # 计算沿梯度上升前后目标函数L(W)的值
        for i in range(len(y_train)):
            tmp = (weight_matrix.T * x_train_matrix[i].T)[0][0]
            mcle_now = mcle_now * i / (i + 1) + (y_train[i] * tmp - np.log(1 + np.exp(tmp))) / (i + 1)
            tmp = (weight_pre.T * x_train_matrix[i].T)[0][0]
            mcle_pre = mcle_pre * i / (i + 1) + (y_train[i] * tmp - np.log(1 + np.exp(tmp))) / (i + 1)
        # 如果两次的目标函数值L(W)之差非常小，就退出
        if np.abs(mcle_now - mcle_pre) < seita:
            break
        else:
            print "L(W) = " + str(mcle_now)
            weight_pre = weight_matrix
        if count > 1000:
            alpha = 1e-3
    return weight_matrix.transpose().tolist()[0], count


# 牛顿法
def newton_method(x_train, y_train):
    num = len(y_train)
    seita = 1e-6
    n = len(x_train[0])                 # 特征的维数 + 1
    count = 0                           # 迭代轮数
    weight_matrix = np.zeros((n, 1))    # 初始化参数
    weight_pre = np.zeros((n, 1))
    x_train_matrix = np.mat(x_train)
    y_train_matrix = np.mat(y_train).T
    while True:
        count = count + 1
        tmp = np.exp(x_train_matrix * weight_matrix) / (1 + np.exp(x_train_matrix * weight_matrix))
        v = []
        for i in range(num):
            array = np.zeros((1, num))[0]
            p = tmp[i][0]
            array[i] = p * (1 - p)
            v.append(array)
        v = np.mat(v)
        # 计算Hessian阵
        h = x_train_matrix.T * v * x_train_matrix
        u = x_train_matrix.transpose() * (tmp - y_train_matrix)
        derta = h.I * u
        weight_matrix = weight_matrix - derta
        mcle_now = 0
        mcle_pre = 0
        # 计算沿梯度上升前后目标函数L(W)的值
        for i in range(len(y_train)):
            tmp1 = (weight_matrix.T * x_train_matrix[i].T)[0][0]
            mcle_now = mcle_now * i / (i + 1) + (y_train[i] * tmp1 - np.log(1 + np.exp(tmp1))) / (i + 1)
            tmp2 = (weight_pre.T * x_train_matrix[i].T)[0][0]
            mcle_pre = mcle_pre * i / (i + 1) + (y_train[i] * tmp2 - np.log(1 + np.exp(tmp2))) / (i + 1)
        # 如果两次的目标函数值L(W)之差非常小，就退出
        if np.abs(mcle_now - mcle_pre) < seita:
            break
        else:
            print "L(W) = " + str(mcle_now)
            weight_pre = weight_matrix
    return weight_matrix.transpose().tolist()[0], count


# 精确率计算，为预测为正例的样本中真正例的个数
def accuracy(x_in, y_in, w_in):
    tp = 0
    fp = 0
    w_matrix = np.mat(w_in)
    for i in range(len(x_in)):
        if (w_matrix * np.mat(x_in[i]).T)[0][0] > 0:
            if y_in[i] == 1:
                tp += 1
            else:
                fp += 1
    return float(tp) / (tp + fp)


# 可选择从文件中读取数据或生成数据，生成数据选项opt：
# 0：生成满足贝叶斯假设，且方差与Y的类别无关的数据
# 1：生成不满足贝叶斯假设，方差与Y的类别无关的数据
# 2：生成满足贝叶斯假设，方差与Y的类别相关的数据
# 3：生成不满足贝叶斯假设，且方差与Y的类别相关的数据
x, y = generate_data(0)
# x, y = load_data()

x1_set = []
for m in range(len(x)):
    x1_set.append(x[m][1])
x_min = min(x1_set)
x_max = max(x1_set)
x_line = np.arange(x_min, x_max, 0.05)
x11 = []
x21 = []
x10 = []
x20 = []
for m in range(len(y)):
    if y[m] is 1:
        x11.append(x[m][1])
        x21.append(x[m][2])
    else:
        x10.append(x[m][1])
        x20.append(x[m][2])

# 通过不同的方法对目标参数矩阵W进行估计
w1, count1 = grad_ascent_without_regular(x, y)
w2, count2 = grad_ascent_with_regular(x, y)
w3, count3 = newton_method(x, y)
print "gradient ascent without regular: ", w1
print "cycle index: " + str(count1)
print "accuracy: " + str(accuracy(x, y, w1))
print "gradient ascent with regular:", w2
print "cycle index: " + str(count2)
print "accuracy: " + str(accuracy(x, y, w2))
print "newton method:", w3
print "cycle index: " + str(count3)
print "accuracy: " + str(accuracy(x, y, w3))
w1_line = -w1[0] / w1[2] - (w1[1] / w1[2]) * x_line
w2_line = -w2[0] / w2[2] - (w2[1] / w2[2]) * x_line
w3_line = -w3[0] / w3[2] - (w3[1] / w3[2]) * x_line

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(x11, x21, linestyle='', marker='.', color='blue', label='Y=1: ' + str(len(x11)))
plt.plot(x10, x20, linestyle='', marker='.', color='red', label='Y=0: ' + str(len(x10)))
plt.plot(x_line, w1_line, label='gradAscent_without_Regular')
plt.plot(x_line, w2_line, label='gradAscent_with_Regular')
plt.legend()

plt.subplot(122)
plt.plot(x11, x21, linestyle='', marker='.', color='blue', label='Y=1:' + str(len(x11)))
plt.plot(x10, x20, linestyle='', marker='.', color='red', label='Y=0:' + str(len(x10)))
plt.plot(x_line, w2_line, label='gradAscent_with_Regular')
plt.plot(x_line, w3_line, label='newton method')
plt.legend()

plt.show()
