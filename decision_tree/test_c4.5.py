import numpy as np
from decision_tree.decision_tree_c import DecisionTree
from decision_tree.treePlotter import createPlot

f = open('train.txt')

lines = f.readlines()
data = [inst.strip().split('\t') for inst in lines]
data = np.asarray(data)

X = np.delete(data, -1, axis=-1)
y = data[:, -1]

features_label = ['age', 'prescript', 'astigmatic', 'tearRate']

tree = DecisionTree(features_label=features_label, method='C4.5')
tree.fit(X, y)

tree_dict = tree._tree
createPlot(tree_dict)


