# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_data():
    data = []
    mean = [20, 40, 30]
    cov = [[5, 0, 0], [0, 5, 0], [0, 0, 0.1]]
    x1, x2, x3 = np.random.multivariate_normal(mean, cov, np.random.randint(60, 80)).T
    for i in range(len(x1)):
        data.append([x1[i], x2[i], x3[i]])
    return data


def load_data():
    x_set = []
    fp = open('data.txt')
    for line in fp.readlines():
        line_array = line.strip().split()
        tmp = []
        for i in range(len(line_array)):
            tmp.append(float(line_array[i]))
        x_set.append(tmp)
    return x_set


# 计算主成分的个数
def compute_number_of_pc(val_list):
    val_sum = sum(val_list)
    tmp = float(0)
    for i in range(len(val_list)):
        tmp += val_list[i]
        if tmp >= val_sum * 0.95:
            return i + 1


def pca(data):
    n = len(data[0])        # 数据的维度
    # 计算各维的均值
    means = []
    for i in range(n):
        tmp = float(0)
        for j in range(len(data)):
            tmp += data[j][i]
        means.append(tmp / len(data))
    # 对所有样本进行中心化
    data_central = []       # m x n, n维维度，m为数据的个数
    for i in range(len(data)):
        tmp = np.mat(data[i]) - np.mat(means)
        data_central.append(tmp.tolist()[0])
    # 计算样本的协方差阵（散度矩阵）
    data_matrix = np.mat(data_central)      # m x n
    cov = data_matrix.T * data_matrix       # n x n 协方差阵
    # 计算协方差阵的特征值与特征向量
    eig_val, eig_vec = np.linalg.eig(cov)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i].T.tolist()[0]) for i in range(len(eig_val))]
    # 按特征值的绝对值降序排列
    eig_pairs.sort(reverse=True)
    # print eig_pairs
    eig_val_list = sorted(eig_val.tolist(), reverse=True)
    # 计算主成分的个数
    # d = 1
    d = compute_number_of_pc(eig_val_list)
    # 选择主成分
    principal_component = []                # d x n
    for i in range(d):
        principal_component.append(eig_pairs[i][1])
    new_data_matrix = data_matrix * np.mat(principal_component).T   # m x d
    return new_data_matrix.tolist(), principal_component


def test_with_data_generated():
    data_set = generate_data()
    new_data_set, pc = pca(data_set)
    print "principal component:"
    print pc
    print "data after dimension decline:"
    print new_data_set

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(221, projection='3d')
    x1_set = [m[0] for m in data_set]
    x2_set = [m[1] for m in data_set]
    x3_set = [m[2] for m in data_set]
    ax.scatter(x1_set, x2_set, x3_set, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax = plt.subplot(222, projection='3d')
    new_x1_set = [m[0] for m in new_data_set]
    new_x2_set = [m[1] for m in new_data_set]
    ax.scatter(new_x1_set, new_x2_set, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.subplot(223)
    plt.plot(new_x1_set, new_x2_set, marker='.', linestyle='')

    plt.show()


test_with_data_generated()
