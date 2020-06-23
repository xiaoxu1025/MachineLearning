import numpy as np
import copy
from decision_tree.cart import BaseTree
"""
ID3/c4.5 的决策树
"""


class DecisionTree(BaseTree):

    methods = ['ID3', 'C4.5']

    def __init__(self, features_label=None, threshold=0.1, method='ID3'):
        self._threshold = threshold
        self.features_label = features_label
        if method not in self.methods:
            raise ValueError('method:{%s} not support' % method)
        self._method = method
        self._tree = None

    def _calc_entropy(self, labels):
        """
        计算熵
        :param labels:
        :return:
        """
        nums = len(labels)
        labels_count = self._labels_freq(labels)
        h_d = 0.
        for label in labels_count:
            p = float(labels_count[label]) / nums
            h_d += -1 * p * np.log2(p)
        return h_d

    def _calc_condition_entropy(self, X, labels):
        unique_features = np.unique(X)
        h_d_a = 0.
        for feature in unique_features:
            prob = float(len(X[X == feature])) / len(X)
            h_d_a += prob * self._calc_entropy(labels[X == feature])
        return h_d_a

    def _get_best_feature(self, X, y):
        n, m = X.shape
        # 最好的特征索引位置
        best_feature = -1
        # 最大信息增益
        max_g_d_a = 0.
        # ID3算法和C4.5除了从信息增益变成信息增益比其他没有变化
        # 得到数据的熵
        h_d = self._calc_entropy(y)
        # 遍历所有的feature
        for i in range(m):
            X_i = X[:, i]
            # ID3算法和C4.5除了从信息增益变成信息增益比其他没有变化
            # 计算条件熵
            h_d_a = self._calc_condition_entropy(X_i, y)
            # 得到信息增益
            g_d_a = h_d - h_d_a
            if self._method == self.methods[1]:
                iv = self._calc_entropy(X_i)
                if iv == 0:
                    continue
                g_d_a = g_d_a / iv
            # 选择信息增益最大的
            if g_d_a > max_g_d_a:
                max_g_d_a = g_d_a
                best_feature = i
        return best_feature, max_g_d_a

    def _split_data(self, X, y, feature, value):
        features = X[:, feature]
        # 得到去掉此特征的数据
        data = np.delete(X, feature, axis=-1)
        indexs = np.where(features == value)[0]
        return data[indexs], y[indexs]

    def _create_tree(self, X, y, features_label):
        labels = list(y)
        # 如果样本中所有样本都属于同一类别则为单节点
        if len(set(labels)) == 1:
            return labels[0]
        # 如果没有特征可以用来划分则直接返回剩下数据中类别数最多的类别
        if len(X[0]) == 0:
            return self._vote_label(y)
        best_feature_index, info_gain = self._get_best_feature(X, y)
        if info_gain < self._threshold:
            return self._vote_label(y)
        unique_features = np.unique(X[:, best_feature_index])
        if features_label is not None:
            best_feature = features_label[best_feature_index]
            del(features_label[best_feature_index])
        else:
            best_feature = best_feature_index
        tree_dict = {best_feature: {}}
        for feature in unique_features:
            sub_X, sub_y = self._split_data(X, y, best_feature_index, feature)
            # 注意进行features_label的拷贝  这样就不会出现index out of range
            # 保证每一次递归用的features_label不是同一个物理地址
            sub_features_label = copy.copy(features_label)
            tree_dict[best_feature][feature] = self._create_tree(sub_X, sub_y, sub_features_label)
        return tree_dict

    def fit(self, X, y):
        features_label = copy.copy(self.features_label)
        self._tree = self._create_tree(X, y, features_label)

    def predict(self, X):
        pass
