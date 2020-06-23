import numpy as np
from decision_tree.cart_classify import DecisionTreeClassify
from decision_tree.treePlotter import createPlot

f = open('train.txt')

lines = f.readlines()
data = [inst.strip().split('\t') for inst in lines]
data = np.asarray(data)

X = np.delete(data, -1, axis=-1)
y = data[:, -1]

features_label = ['age', 'prescript', 'astigmatic', 'tearRate']

tree = DecisionTreeClassify(features_label=features_label)
tree.fit(X, y)
tree._last_order(tree._tree)
# tree_dict = tree._tree
# createPlot(tree_dict)

