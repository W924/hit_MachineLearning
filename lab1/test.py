import numpy as np
import random
import matplotlib.pyplot as plt

N = 14
order = 9
lamda = 1e-5
seita = 0.00001
alpha = 0.01

X = np.linspace(0, 1, N, endpoint=True)
Y = np.sin(2 * np.pi * X)
T = []
for i in range(len(Y)):
    T.append(Y[i] + random.gauss(0, 0.5))
T = np.transpose(np.mat(T))

X_Matrix = []
for i in range(X.size):
    tmp = []
    for j in range(order + 1):
        tmp.append(np.power(X[i], j))
    X_Matrix.append(tmp)
X_Matrix = np.mat(X_Matrix)


def analyse_with_regular(x_in, t_in, lamda_in):
    unit_matrix = np.eye(order + 1, dtype=float)
    return (x_in.T * x_in + lamda_in * unit_matrix).I * x_in.T * t_in


def compute_polynomial(x_in, w):
    t = []
    w_in = w.transpose().tolist()[0]
    for m in range(x_in.size):
        add = 0.0
        for n in range(len(w_in)):
            add += w_in[n] * np.power(x_in[m], n)
        t.append(add)
    return t


def loss(w, x_in, y_in):
    y_tran = compute_polynomial(x_in, w)
    loss_sum = 0
    for m in range(len(y_tran)):
        loss_sum += (y_tran[m] - y_in[m, 0]) * (y_tran[m] - y_in[m, 0])
    return np.power((loss_sum / x_in.size), 0.5)


X_Test = np.linspace(0, 1, 4)
Y_Test = np.sin(2 * np.pi * X_Test)
T_Test = []
for i in range(len(X_Test)):
    T_Test.append(Y_Test[i] + random.gauss(0, 0.5))
    print Y_Test[i]
    print T_Test[i]
T_Test = (np.mat(T_Test)).T

e_lamda = np.arange(-20, -5, 1)
loss_set = []
for i in range(len(e_lamda)):
    W = analyse_with_regular(X_Matrix, T, e_lamda[i])
    loss_set.append(loss(W, X_Test, T_Test))
print loss_set

plt.plot(e_lamda, loss_set, linestyle='', marker=".")
plt.show()





