import numpy as np
from decision_tree.cart import BaseTree
from copy import copy

"""
CART 分类回归树
"""


class DecisionTreeRegression(BaseTree):

    def __init__(self, features_label=None, threshold=0.1):
        self._features_label = features_label
        self._threshold = threshold
        self._tree = None

    def _mean(self, y):
        return np.mean(y)

    def _mean_square(self, y):
        """
        计算均方误差
        :param y:
        :return:
        """
        return np.var(y) * len(y)

    def _split_data(self, X, y, feature, value):
        # 返回数据满足条件数据所在位置
        data = np.delete(X, feature, axis=-1)
        X_i = X[:, feature]
        left_ids = np.where(X_i <= value)[0]
        right_ids = np.where(X_i > value)[0]
        return data[left_ids], data[right_ids], y[left_ids], y[right_ids]

    def _get_best_feature(self, X, y):
        n, m = X.shape
        best_feature_index = -1
        best_split_value = -1
        best_mean_square_error = np.inf
        # 遍历所有特征
        for i in range(m):
            X_i = X[i]
            unique_X_i = np.unique(X_i)
            if len(unique_X_i) == 1:
                continue
            for value in unique_X_i:
                _, _, left_ids, right_ids = self._split_data(X, y, i, value)
                mean_square_error = self._mean_square(y[left_ids]) + self._mean_square(y[right_ids])
                if best_mean_square_error > mean_square_error:
                    best_mean_square_error = mean_square_error
                    best_feature_index = i
                    best_split_value = value
        return best_feature_index, best_split_value, best_mean_square_error

    def _create_tree(self, X, y, features_label):
        if len(X[0]) == 0:
            return self._mean(y)
        best_feature_index, best_split_value, min_error = self._get_best_feature(X, y)
        # 样本无法差分
        if best_feature_index == -1:
            return self._mean(y)
        if min_error <= self._threshold:
            return self._mean(y)
        if features_label is not None:
            best_feature = features_label[best_feature_index]
            best_feature = '%s-%s' % (best_feature, best_split_value)
            del (features_label[best_feature_index])
        else:
            best_feature = '%s-%s' % (best_feature_index, best_split_value)
        tree_dict = {best_feature: {}}
        # 得到左右区域数据
        left_X, right_X, left_y, right_y = self._split_data(X, y, best_feature_index, best_split_value)
        features_label_left = copy(features_label)
        features_label_right = copy(features_label)
        tree_dict[best_feature]['left'] = self._create_tree(left_X, left_y, features_label_left)
        tree_dict[best_feature]['right'] = self._create_tree(right_X, right_y, features_label_right)
        return tree_dict

    def fit(self, X, y):
        features_label = copy(self._features_label)
        self._tree = self._create_tree(X, y, features_label)

    def predict(self, X):
        pass
