import numpy as np

class LogisticRegressionClassfiy:

    def __init__(self, lr=.001, iter=500):
        self.lr = lr
        self.iter = iter
        self.W = None

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def fit(self, X, y):
        n, m = X.shape
        W = np.zeros((m, 1), dtype=np.float32)
        y = np.reshape(y, (n, 1))
        for i in range(self.iter):
            # (n, 1)
            h = self.sigmoid(np.dot(X, W))
            W = W - self.lr * np.dot(X.T, (h - y))
        self.W = W

    def score(self, X_test, y_test):
        right = 0
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.W)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)
