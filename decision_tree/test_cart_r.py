import numpy as np
from decision_tree.cart_regression import DecisionTreeRegression
from decision_tree.treePlotter import createPlot

f = open('train2.txt')

lines = f.readlines()
data = [[int(str) for str in inst.strip().split('\t')] for inst in lines]
data = np.asarray(data)


X = np.delete(data, -1, axis=-1)
y = data[:, -1]

features_label = ['age', 'prescript', 'astigmatic', 'tearRate']

tree = DecisionTreeRegression(features_label=features_label)
tree.fit(X, y)


tree_dict = tree._tree
createPlot(tree_dict)
