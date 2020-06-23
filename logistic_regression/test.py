import numpy as np
from logistic_regression.lr import LogisticRegressionClassfiy
from logistic_regression.load_data import PreProcess
import matplotlib.pyplot as plt


def main():
    process = PreProcess('./data.txt')
    X, y = process.load_data()
    model = LogisticRegressionClassfiy()
    model.fit(X, y)
    weights = model.W
    # 对两类数据进行分类
    x1_red, x2_red = [], []
    x1_blue, x2_blue = [], []
    nums = len(X)
    for i in range(nums):
        if int(y[i]) == 1:
            x1_red.append(X[i, 1])
            x2_red.append(X[i, 2])
        else:
            x1_blue.append(X[i, 1])
            x2_blue.append(X[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1_red, x2_red, s=30, c='red', marker='s')
    ax.scatter(x1_blue, x2_blue, s=30, c='blue', marker='s')
    print(weights)
    # 分界线
    # 取概率=0.5,即线性计算为0时对应的点
    # sigmoid(WX) = 0.5
    # 0 = w0x0 + w1x1 + w2x2
    # x2 = (-w0 - w1*x1) / w2
    x1 = np.arange(-5.0, 5.0, 0.1)
    x2 = (-float(weights[0][0]) - float(weights[1][0]) * x1) / float(weights[2][0])

    ax.plot(x1, x2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    main()
