import numpy as np
from decision_tree.cart import BaseTree
from itertools import combinations
from copy import copy

"""
CART 分类树
"""


class DecisionTreeClassify(BaseTree):

    def __init__(self, features_label=None, threshold=0.01):
        self._features_label = features_label
        self._threshold = threshold
        self._tree = None

    def _calc_gini(self, y):
        label_count = {}
        for label in y:
            if label not in label_count:
                label_count.setdefault(label, 0)
            label_count[label] += 1
        gini = 1.
        data_num = len(y)
        for label in label_count:
            value = label_count[label]
            prop = value / data_num
            gini -= prop ** 2
        return gini

    def _split_gather(self, features):
        feature_num = len(features)
        if feature_num < 2:
            raise ValueError('split features len should > 2, please check %s' % features)
        gathers = []
        for i in range(1, feature_num):
            gather = list(combinations(features, len(features[0:i])))
            gathers.extend(gather)
        gather_num = len(gathers)
        mid_idx = int(gather_num / 2)
        return zip(gathers[0:mid_idx], gathers[mid_idx:][::-1])

    def _get_best_feature(self, X, y):
        n, m = X.shape
        best_feature_index = -1
        best_left = []
        best_right = []
        best_left_ids = []
        best_right_ids = []
        best_gini = np.inf
        # 遍历所有特征
        for i in range(m):
            X_i = X[:, i]
            unique_X_i = np.unique(X_i)
            if len(unique_X_i) < 1:
                continue
            split_gather = self._split_gather(unique_X_i)
            for left, right in split_gather:
                gini = 0.
                left_ids = []
                for left_value in left:
                    left_ids.extend(np.where(X_i == left_value)[0])
                # 计算拆分后的左增益
                left_prob = len(left_ids) / n
                gini += left_prob * self._calc_gini(y[left_ids])
                # 计算右增益
                right_ids = []
                for right_value in right:
                    right_ids.extend(np.where(X_i == right_value)[0])
                right_prob = len(right_ids) / n
                gini += right_prob * self._calc_gini(y[right_ids])
                if best_gini > gini:
                    best_gini = gini
                    best_feature_index = i
                    best_left_ids = left_ids
                    best_right_ids = right_ids
                    best_left = left
                    best_right = right
        return best_feature_index, best_left_ids, best_right_ids, best_left, best_right, best_gini

    def _split_data(self, X, y, feature, left_ids, right_ids):
        data = np.delete(X, feature, axis=-1)
        return data[left_ids], data[right_ids], y[left_ids], y[right_ids]

    def _create_tree(self, X, y, features_label):
        labels = list(y)
        # 如果样本中所有样本都属于同一类别则为单节点
        if len(set(labels)) == 1:
            return labels[0]
        # 如果没有特征可以用来划分则直接返回剩下数据中类别数最多的类别
        if len(X[0]) == 0:
            return self._vote_label(y)
        best_feature_index, best_left_ids, best_right_ids, best_left, best_right, best_gini = self._get_best_feature(X, y)
        if best_gini < self._threshold:
            return self._vote_label(y)
        if features_label is not None:
            best_feature = features_label[best_feature_index]
            best_feature = '%s/%s' % (best_left, best_right)
            del (features_label[best_feature_index])
        else:
            best_feature = '%s/%s' % (best_left, best_right)
        tree_dict = {best_feature: {}}
        # 得到左右区域数据
        left_X, right_X, left_y, right_y = self._split_data(X, y, best_feature_index, best_left_ids, best_right_ids)
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

    # 这里是对分类回归树进行减枝
    def prune(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """
        self._tree_data = {}
        # 1 为了方便计算 先将X存入树中


        # 2 建立子树序列
        T = []

    def _last_order(self, node, gts):
        """
        后续遍历树
        :param node:
        :param gts 用来存储子树的所有gt 然后可以求出最佳子树
        :return:
        """
        if self._is_leaf(node):
            return
        # 这里只有一个节点 for只会遍历一次 每次拆分都是根据一个属性节点进行split
        # for key, value in node.items():
        #     left_node = value['left']
        #     right_node = value['right']

        key = list(node.keys())[0]
        self._last_order(node[key]['left'])
        self._last_order(node[key]['right'])
        # 处理节点
        X, y = self._tree_data[key]
        # 这里用gini表示划分的好坏
        Ct = self._calc_gini(y)
        _, _, _, _, _, CT = self._get_best_feature(X, y)
        gt = (Ct - CT) / self._leaf_num(node)
        gts.append(gt)