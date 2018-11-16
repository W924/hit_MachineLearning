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
    fp = open('data.txt')
    for line in fp.readlines():
        line_array = line.strip().split(',')
        tmp = []
        for i in range(len(line_array)):
            tmp.append(float(line_array[i]))
        x_set.append(tmp)
    return x_set


# 选择初始均值向量
def initial_point(data, k):
    n = len(data[0])
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
                temp = float(0)
                for c in range(n):
                    temp += np.square(point[c] - mean[c])
                temp = np.sqrt(temp)
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
    n = len(data[0])
    means = initial_point(data, k)
    label = []                           # 标签集合
    for i in range(len(data)):
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
                temp = float(0)
                for c in range(n):
                    temp += np.square(data[i][c] - means[j][c])
                temp = np.sqrt(temp)
                if temp < distance:
                    distance = temp
                    sample_label = j
            label[i] = sample_label
            divide[sample_label].append(data[i])

        flag = True
        # 重新计算各类的新均值向量
        new_means = []
        for i in range(k):
            mean_new = []
            for c in range(n):
                tmp_sum = float(0)
                for j in range(len(divide[i])):
                    tmp_sum += divide[i][j][c]
                tmp_mean = tmp_sum / len(divide[i])
                if tmp_mean != means[i][c]:
                    flag = False
                mean_new.append(tmp_mean)
            new_means.append(mean_new)
        means = new_means
        # 如果均值向量未更新，则退出
        if flag is True:
            data_divide = divide
            # print "k_means:", means
            break
    return means, data_divide


# 计算多维高斯概率
def gauss_probability(x_in, mean_in, cov_in):
    dimension = len(x_in[0])           # 维度
    tmp = float(-0.5 * (x_in - mean_in).T * cov_in.I * (x_in - mean_in))
    div = pow(2 * np.pi, dimension/2) * pow(np.linalg.det(cov_in), 0.5)
    return pow(np.e, tmp) / div


# 计算当前似然函数的值
def lln(gamma_in, alpha_in, k):
    likehood = float(0)
    for i in range(len(gamma_in)):
        tmp = float(0)
        for j in range(k):
            tmp += alpha_in[j] * gamma_in[i][j]
        likehood += np.log(tmp)
    return likehood


# EM算法求解高斯混合聚类
def gmm(data, k, means_in):
    n = len(data[0])
    alpha = []
    # means = initial_point(data, k)
    means = means_in        # 使用k_means的结果作为均值初始向量
    covs = []
    # 初始化
    for i in range(k):
        alpha.append(1.0 / k)
        covs.append(np.eye(n).tolist())
    pre_mle = float(0)
    count = 0
    while True:
        # 计算每个x各类的后验概率
        count += 1
        gamma = []
        for i in range(len(data)):
            tmp = []
            probability_sum = float(0)
            for j in range(k):
                pro = alpha[j] * gauss_probability(np.mat(data[i]).T, np.mat(means[j]).T, np.mat(covs[j]))
                tmp.append(pro)
                probability_sum += pro
            for j in range(k):
                tmp[j] = tmp[j] / probability_sum
            gamma.append(tmp)
        mle = lln(gamma, alpha, k)
        print mle
        # 如果两次迭代似然函数基本不变或者迭代足够多的次数，则停止迭代
        if np.abs(mle - pre_mle) < 1e-4:
            break
        if count > 200:
            break
        pre_mle = mle
        # 计算新的混合系数、均值和方差
        new_alpha = []
        new_means = []
        new_covs = []
        for i in range(k):
            gamma_sum = float(0)          # 该类的后验概率之和
            for j in range(len(data)):
                gamma_sum += gamma[j][i]
            new_alpha.append(gamma_sum / len(data))    # 新的混合系数
            tmp = []
            for c in range(n):
                tmp_sum = float(0)
                for j in range(len(data)):
                    tmp_sum += gamma[j][i] * data[j][c]
                tmp.append(tmp_sum)
            new_mean = [e / gamma_sum for e in tmp]
            new_means.append(new_mean)  # 新的均值向量
            tmp = np.zeros((n, n))
            for j in range(len(data)):
                tmp += gamma[j][i] * (np.mat(data[j]).T - np.mat(new_mean).T) * (np.mat(data[j]).T - np.mat(new_mean).T).T
            new_cov = tmp / gamma_sum   # 新的协方差阵
            new_covs.append(new_cov.tolist())
        alpha = new_alpha
        means = new_means
        covs = new_covs
    divide = [[0 for col in range(0)] for row in range(k)]
    for i in range(len(data)):
        max_label = 0
        max_pro = 0
        for j in range(k):
            if gamma[i][j] > max_pro:
                max_label = j
                max_pro = gamma[i][j]
        divide[max_label].append(data[i])
    return divide, means


_k = 6                               # k_means算法中的k值
data_set = generate_data(6)          # 生成m个高斯分布的数据
# data_set = load_data()
means_by_kmeans, divide_set_by_kmeans = k_means(data_set, _k)
divide_set_by_gmm, means_by_gmm = gmm(data_set, _k, means_by_kmeans)

print "\nmeans(k-means):"
print means_by_kmeans
print "means(gmm):"
print means_by_gmm

plt.figure(figsize=(10, 5))
plt.subplot(121)
for m in range(_k):
    y = divide_set_by_kmeans[m]
    x1_set = []
    x2_set = []
    for n in range(len(y)):
        x1_set.append(y[n][0])
        x2_set.append(y[n][1])
    plt.plot(x1_set, x2_set, marker='.', linestyle='')
plt.title("k-means")
plt.legend()

plt.subplot(122)
for m in range(_k):
    y = divide_set_by_gmm[m]
    x1_set = []
    x2_set = []
    for n in range(len(y)):
        x1_set.append(y[n][0])
        x2_set.append(y[n][1])
    plt.plot(x1_set, x2_set, marker='.', linestyle='')
plt.title("GMM")
plt.legend()

plt.show()


# # 在实际数据集上进行测试
# _k = 3
# data_set = load_data()
# means_by_kmeans, divide_set_by_kmeans = k_means(data_set, _k)
# divide_set_by_gmm, means_by_gmm = gmm(data_set, _k, means_by_kmeans)
#
# print "\nmeans(k-means):"
# print means_by_kmeans
# print "means(gmm):"
# print means_by_gmm
