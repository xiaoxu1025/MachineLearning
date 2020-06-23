"""
所有树的父类
实现了减枝处理
只需要继承该类 所有树都能进行减枝操作
"""


class BaseTree:

    def __init__(self):
        pass

    def _labels_freq(self, labels):
        """
        统计labels频次
        :param labels:
        :return:
        """
        labels_count = {}
        for label in labels:
            if label not in labels_count.keys():
                labels_count[label] = 0
            labels_count[label] += 1
        return labels_count

    def _vote_label(self, labels):
        """
        投票选出最大类别
        :param labels:
        :return:
        """
        labels_count = self._labels_freq(labels)
        # 对label进行降序排序
        labels_sort = sorted(labels_count.items(), key=lambda item: item[1], reverse=True)
        return labels_sort[0][0]

    def _is_leaf(self, node):
        """
        该节点是否是叶子节点
        :param node:
        :return:
        """
        return type(node) != dict

    def _is_tree(self, node):
        return type(node) == dict

    def _leaf_num(self, node):
        """
        叶子节点数量
        :param node:
        :return:
        """
        if self._is_leaf(node):
            return 1
        for key, value in node.items():
            left_node = value['left']
            right_node = value['right']
        return self._leaf_num(left_node) + self._leaf_num(right_node)