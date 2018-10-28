# coding:utf-8
import numpy as np
import random
import matplotlib.pyplot as plt

# 初始化参数
N = 10
order = 9
lamda = np.exp(-7)
seita = 0.00001
alpha = 0.01

# 生成均匀的训练样本，并加入随机噪声
X = np.linspace(0, 1, N, endpoint=True)
Y = np.sin(2 * np.pi * X)
T = []
for i in range(len(Y)):
    T.append(Y[i] + random.gauss(0, 0.4))
T = np.transpose(np.mat(T))

# 计算X的矩阵
X_Matrix = []
for i in range(X.size):
    tmp = []
    for j in range(order + 1):
        tmp.append(np.power(X[i], j))
    X_Matrix.append(tmp)
X_Matrix = np.mat(X_Matrix)


# 解析法（最小二乘法），不加正则项
def analyse_without_regular(x_in, t_in):
    return (x_in.T * x_in).I * x_in.T * t_in


# 解析法，加入正则项
def analyse_with_regular(x_in, t_in, lamda_in):
    unit_matrix = np.eye(order + 1, dtype=float)
    return (x_in.T * x_in + lamda_in * unit_matrix).I * x_in.T * t_in


# 梯度公式
def gradient(x_in, w_in, t_in, lamda_in):
    return (x_in.T * x_in * w_in - x_in.T * t_in + lamda_in * w_in) / len(x_in)


# 梯度下降法
def gradient_descent(x_in, t_in, lamda_in):
    w_out = np.zeros((order + 1, 1))
    k = 0
    while True:
        k += 1
        distance = alpha * gradient(x_in, w_out, t_in, lamda_in)
        flag = True
        for m in range(order + 1):
            if np.abs(distance[m, 0]) >= seita:
                flag = False
                break
        if flag is True:
            print "k = ", k, ", gradient descent: seita = ", seita, ", alpha = ", alpha, ", lamda = ", lamda_in
            break
        else:
            w_out = w_out - distance
    return w_out


# 共轭梯度法
def conjugate_gradient(x_in, t_in, lamda_in):
    x_out = np.zeros((order + 1, 1))
    A = x_in.T * x_in + lamda_in * np.eye(order + 1, dtype=float)
    b = x_in.T * t_in
    r = b - A * x_out
    p = r
    k = 0
    while True:
        a = ((r.T * r) / (p.T * A * p))[0, 0]
        x_out = x_out + a * p
        r_1 = r - a * A * p
        if (r_1.T * r_1)[0, 0] <= seita:
            break
        else:
            beta = ((r_1.T * r_1) / (r.T * r))[0, 0]
            p = r_1 + beta * p
            k = k + 1
            r = r_1
    print "k = ", k, ", conjugate gradient descent: seita = ", seita, " ,lamda = ", lamda_in
    return x_out


# 多项式函数，给定X集合，输出Y的集合
def compute_polynomial(x_in, w):
    t = []
    w_in = w.transpose().tolist()[0]
    for m in range(x_in.size):
        add = 0.0
        for n in range(len(w_in)):
            add += w_in[n] * np.power(x_in[m], n)
        t.append(add)
    return t


# 损失函数，定义为均匀平差的二次方根
def loss(w, x_in, y_in):
    y_tran = compute_polynomial(x_in, w)
    loss_sum = 0.0
    for m in range(len(y_tran)):
        loss_sum += (y_tran[m] - y_in[m, 0]) * (y_tran[m] - y_in[m, 0])
    return np.power((loss_sum / x_in.size), 0.5)


# 在同一训练集上，通过四种方法进行训练
W_1 = analyse_without_regular(X_Matrix, T)
print W_1.T
W_2 = analyse_with_regular(X_Matrix, T, lamda)
W_3 = gradient_descent(X_Matrix, T, lamda)
W_4 = conjugate_gradient(X_Matrix, T, lamda)

# 计算四种方法得到的模型在训练集上的loss
x = np.arange(0, 1, 0.005)
y = np.sin(2 * np.pi * x)
y_w1 = compute_polynomial(x, W_1)
loss_1 = loss(W_1, X, T)
print "loss of training data(analyse method without regular) : ", loss_1
y_w2 = compute_polynomial(x, W_2)
loss_2 = loss(W_2, X, T)
print "loss of training data(analyse method with regular) : ", loss_2
y_w3 = compute_polynomial(x, W_3)
loss_3 = loss(W_3, X, T)
print "loss of training data(gradient descent) : ", loss_3
y_w4 = compute_polynomial(x, W_4)
loss_4 = loss(W_4, X, T)
print "loss of training data(conjugate gradient) : ", loss_4

# 生成验证集，过程与训练集相同
N_Test = 4
X_Test = np.linspace(0, 1, N_Test)
Y_Test = np.sin(2 * np.pi * X_Test)
T_Test = []
for i in range(len(Y_Test)):
    T_Test.append(Y_Test[i] + random.gauss(0, 0.4))
T_Test = np.transpose(np.mat(T_Test))

# 通过不同的lamda进行训练，找到在验证机上loss最小的lamda
e_lamda = np.arange(-50, 0, 1)
loss_train = []
loss_test = []
for i in range(len(e_lamda)):
    w_train = analyse_with_regular(X_Matrix, T, np.exp(e_lamda[i]))
    loss_train.append(loss(w_train, X, T))
    loss_test.append(loss(w_train, X_Test, T_Test))

# 使用matplotlib绘制四种方法的结果和不同lamda的loss图像
N_label = "N = " + str(N)
Order_label = "M = " + str(order)
lamda_label = "lamda = " + str(lamda)
NTest_label = "N_Test = " + str(N_Test)

plt.figure(figsize=(10, 10))
plt.subplot(321)
plt.plot(x, y, label="y=sin(2*pi*x)")
plt.plot(X, T, linestyle='', marker='.')
plt.plot(x, y_w1, label="analyse without regular")
plt.title(N_label + "  " + Order_label)
plt.legend()

plt.subplot(322)
plt.plot(x, y, label="y=sin(2*pi*x)")
plt.plot(X, T, linestyle='', marker='.')
plt.plot(x, y_w2, label="analyse with regular")
plt.title(N_label + "  " + Order_label + "  " + lamda_label)
plt.legend()

plt.subplot(323)
plt.plot(x, y, label="y=sin(2*pi*x)")
plt.plot(X, T, linestyle='', marker='.')
plt.plot(x, y_w3, label="gradient descent")
plt.title(N_label + "  " + Order_label + "  " + lamda_label)
plt.legend()

plt.subplot(324)
plt.plot(x, y, label="y=sin(2*pi*x)")
plt.plot(X, T, linestyle='', marker='.')
plt.plot(x, y_w4, label="conjugate gradient")
plt.title(N_label + "  " + Order_label + "  " + lamda_label)
plt.legend()

plt.subplot(325)
plt.plot(e_lamda, loss_train, label="Training")
plt.plot(e_lamda, loss_test, label="Test")
plt.ylabel("ERMS")
plt.xlabel("lnlamda")
plt.title(N_label + "  " + Order_label + "  " + NTest_label)
plt.legend()

plt.show()
