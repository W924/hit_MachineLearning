# coding:utf-8
import numpy as np
import random
import matplotlib.pyplot as plt


# 随机生成k个高斯分布的数据
def generate_data(k):
    data = []
    cov = [[1, 0], [0, 1]]
    for i in range(k):
        x1_mean = random.randint(0, 8)
        x2_mean = random.randint(0, 8)
        mean = [x1_mean, x2_mean]
        x1, x2 = np.random.multivariate_normal(mean, cov, np.random.randint(25, 30)).T
        for j in range(len(x1)):
            data.append([x1[j], x2[j]])
    return data


# 从文件中读取数据
def load_data():
    x_set = []
    fp = open('data1.txt')
    for line in fp.readlines():
        line_array = line.strip().split()
        x_set.append([float(line_array[0]), float(line_array[1])])
    return x_set


# 选择初始均值向量
def initial_point(data, k):
    means_list = []
    mean_rand = random.sample(data, 1)[0]  # 随机选择一个点作为第一个中心点
    means_list.append(mean_rand)
    for i in range(k-1):                # 找剩余的k-1个中心点
        ret = []                        # 非中心点的列表
        for j in data:
            if j not in means_list:
                ret.append(j)
        max_distance = float('-inf')    # 非中心点到中心点的最短距离的最大值
        max_point = None
        for point in ret:
            # print point
            distance = float('inf')     # 该点到中心点的最短距离
            # 计算该点到中心点的最短距离
            for mean in means_list:
                # print mean
                temp = np.sqrt(np.square(point[0] - mean[0]) + np.square(point[1] - mean[1]))
                if temp < distance:
                    distance = temp
            # 找每个非中心点到中心点最短距离的最大值
            if max_distance < distance:
                max_distance = distance
                max_point = point
        means_list.append(max_point)
        # print max_point
    return means_list


# k-means算法
def k_means(data, k):
    # means = random.sample(data, k)       # 算法开始时随机选择k个样本作为初始均值向量
    means = initial_point(data, k)
    label = []                           # 标签集合
    for n in range(len(data)):
        label.append(0)
    # data_divide = []
    while True:
        divide = [[0 for col in range(0)] for row in range(k)]
        # 根据当前各类的中心点，对每个数据贴标签
        for i in range(len(data)):
            distance = float('inf')
            sample_label = 0
            # 确定最近的中心点
            for j in range(len(means)):
                temp = np.sqrt(np.square(data[i][0] - means[j][0]) + np.square(data[i][1] - means[j][1]))
                if temp < distance:
                    distance = temp
                    sample_label = j
            label[i] = sample_label
            divide[sample_label].append(data[i])
        flag = True
        # 重新计算各类的新均值向量
        for i in range(k):
            x1_sum = float(0)
            x2_sum = float(0)
            for j in range(len(divide[i])):
                x1_sum += divide[i][j][0]
                x2_sum += divide[i][j][1]
            # 新的均值向量
            x1_mean = x1_sum / len(divide[i])
            x2_mean = x2_sum / len(divide[i])
            if (x1_mean != means[i][0]) | (x2_mean != means[i][1]):
                flag = False
            means[i] = [x1_mean, x2_mean]
        # 如果均值向量未更新，则退出
        if flag is True:
            data_divide = divide
            break
    return label, data_divide


_k = 6                               # k_means算法中的k值
data_set = generate_data(6)          # 生成m个高斯分布的数据
label_set, divide_set = k_means(data_set, _k)
for m in range(_k):
    y = divide_set[m]
    x1_set = []
    x2_set = []
    for n in range(len(y)):
        x1_set.append(y[n][0])
        x2_set.append(y[n][1])
    plt.plot(x1_set, x2_set, marker='.', linestyle='')
plt.show()
