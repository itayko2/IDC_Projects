import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}


def class_counts(data):
    """Counts the number of each type of target value"""
    counts = {1.0: 0, 0.0: 0}  # a dictionary of label -> count.
    for row in data:
        label = row[-1]
        counts[(label == 1)] += 1
    return counts


def num_of_features(data):
    return np.size(data, 1) - 1


def convert(result):
    if result[1.0] >= result[0.0]:
        return 1
    else:
        return 0


def unique_vals(data, col):
    """unique values column"""
    vals = ([row[col] for row in data])
    return np.sort(vals)


def threshold_vals(data, col):
    vals = []
    regular_vals = unique_vals(data, col)
    for i in range(len(regular_vals) - 1):
        vals.append((regular_vals[i] + regular_vals[i + 1]) / 2)
    return vals


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    counts = class_counts(data)
    gini = 1
    if len(data) == 0:
        return 0
    for label in counts:
        prob_of_label = counts[label] / float(len(data))
        gini -= prob_of_label**2
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    counts = class_counts(data)
    entropy = 0
    for label in counts:
        prob_of_label = counts[label] / float(len(data))
        if prob_of_label == 0:
            continue
        entropy -= prob_of_label * np.log2(prob_of_label)
    return entropy


def info_gain(current_impurity, impurity, true_data, false_data):
    """Calculate the info gain"""
    p = float(len(true_data)) / (len(false_data) + len(true_data))
    return current_impurity - p * impurity(true_data) - (1 - p) * impurity(false_data)


def question_match(data, row, feature, value):
    return data[row][feature] > value


def question_match_node(node, instance):
    return instance[node.feature] > node.value


def partition(data, feature, value):
    """Divide the data to true_data and false_data according to asked question."""

    true_data, false_data = [], []
    for i in range(len(data)):
        if question_match(data, i, feature, value):
            true_data.append(data[i])
        else:
            false_data.append(data[i])
    return true_data, false_data


def find_best_split(data, impurity):
    """Find the best feature and value to ask by iterating over every feature and threshold value."""
    best_gain = 0
    best_feature = 0
    best_value = 0
    current_impurity = impurity(data)
    n_features = num_of_features(data)

    for col in range(n_features):

        values = threshold_vals(data, col)

        for value in values:  # for each threshold value.

            # partitioning the data
            true_data, false_data = partition(data, col, value)

            # Skip this split if it doesn't divide the dataset.
            if len(true_data) == 0 or len(false_data) == 0:
                continue

            # Calculate the info gain from this split
            gain = info_gain(current_impurity, impurity, true_data, false_data)

            if gain > best_gain:
                best_gain, best_feature, best_value = gain, col, value

    return best_gain, best_feature, best_value


def chi_square(node, data):
    """Calculate chi square"""
    chi = 0
    true_data, false_data = partition(data, node.feature, node.value)
    p_false = len(false_data) / (len(data))
    p_true = 1 - p_false
    d0 = len(false_data)
    d1 = len(true_data)
    p0 = class_counts(false_data)[0.0]
    n0 = class_counts(false_data)[1.0]
    p1 = class_counts(true_data)[0.0]
    n1 = class_counts(true_data)[1.0]
    chi += (np.square(p0 - (d0 * p_false))) / (d0 * p_false) + (np.square(n0 - (d0 * p_true))) / (d0 * p_true)
    chi += (np.square(p1 - (d1 * p_false))) / (d1 * p_false) + (np.square(n1 - (d1 * p_true))) / (d1 * p_true)
    return chi


class LeafNode:
    """Node is a leaf"""

    def __init__(self, data):
        self.children = []
        self.data = class_counts(data)  # dict with a data(count true , count false)


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.

    def __init__(self, feature, value, gain, data):
        self.feature = feature  # column index of criteria being tested
        self.value = value  # value necessary to get a true result
        self.gain = gain
        self.data = class_counts(data)  # dict with a data(count true , count false)
        self.children = []

    def add_child(self, node):
        self.children.append(node)


def build_tree(data, impurity, chi=0):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    node_gain, node_feature, node_value = find_best_split(data, impurity)
    root = DecisionNode(node_feature, node_value, node_gain, data)
    # Base case: returning a leaf with 0 gain ( perfect splitting)
    if root.gain == 0:
        return LeafNode(data)


    true_data, false_data = partition(data, root.feature, root.value)
    if chi > chi_square(root, data):
        return LeafNode(data)
    root.add_child(build_tree(true_data, impurity, chi))
    root.add_child(build_tree(false_data, impurity, chi))

    return root


def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    # Base case: leaf
    if len(node.children) == 0:
        return convert(node.data)

    # ask the question on the instance
    if question_match_node(node, instance):
        return predict(node.children[0], instance)
    else:
        return predict(node.children[1], instance)


def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    for instance in dataset:
        prediction = predict(node, instance)
        if prediction == instance[-1]:
            accuracy += 1

    return accuracy / len(dataset)


def list_of_parents(node):
    """Calculate all parents with a leaf child"""
    if len(node.children[0].children) == 0 and len(node.children[1].children) == 0:
        return [node]

    elif len(node.children[0].children) == 0:
        return [node]+list_of_parents(node.children[1])

    elif len(node.children[1].children) == 0:
        return [node]+list_of_parents(node.children[0])

    return list_of_parents(node.children[0])+list_of_parents(node.children[1])


def number_of_internal_nodes(root):
    """Calculate internal nodes"""
    if len(root.children) == 0:
        return 0
    return number_of_internal_nodes(root.children[0]) + number_of_internal_nodes(root.children[1]) + 1


def post_pruning(root, dataset):
    """Post pruning"""
    best_node = 0
    best_accuracy = 0
    parents = list_of_parents(root)
    for parent in parents:
        right_child = parent.children[0]
        left_child = parent.children[1]
        del parent.children[1]
        del parent.children[0]
        current_accuracy = calc_accuracy(root, dataset)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_node = parent
        parent.add_child(right_child)
        parent.add_child(left_child)

    del best_node.children[1]
    del best_node.children[0]


def print_tree(node, space=""):
    """
    prints the tree according to the example in the notebook

    Input:
    - node: a node in the decision tree

    This function has no return value
    """

    # Base case: leaf
    if len(node.children) == 0:
        if node.data[1.0] != 0:
            print (space + "leaf: [{1.0 : "+str(node.data[1.0])+"}]")
        if node.data[0.0] != 0:
            print (space + "leaf: [{0.0 : "+str(node.data[0.0])+"}]")
        return

    # Print the feature and the value
    print (space + "[" + str(node.feature) + " <= " + str(node.value) + "],")

    print_tree(node.children[1], (space + "  "))
    print_tree(node.children[0], (space + "  "))


