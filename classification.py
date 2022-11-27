#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed.
##############################################################################

import numpy as np

# Leaf node, bottom of the tree, no more data split after this point
class Leaf_Node:
    def __init__(self, labels, depth):
        self.label = max(set(labels), key=labels.tolist().count)
        self.depth = depth

    def __repr__(self):
        return f"Node({self.depth}) [Label: {self.label}]"

    def predict(self, instance):
        return self.label


# Decision node, contains rule on how to split data, and reference to
class Decision_Node:
    def __init__(self, attribute, value, left, right, depth):
        self.left = left
        self.right = right
        self.attribute = attribute
        self.value = value
        self.depth = depth

    def __repr__(self):
        tab = "\t"
        return f"Node({self.depth}) [Attribute: {self.attribute} Value: {self.value}] \
      \n {tab * self.depth} L -> {self.left} \
      \n {tab * self.depth} R -> {self.right}"

    def predict(self, instance):
        if instance[self.attribute] < self.value:
            return self.left.predict(instance)
        else:
            return self.right.predict(instance)


def split_dataset(instances, labels, i):
    """Split a dataset on a given index
    Args:
    instances (numpy.ndarray): Instances, numpy array of shape (N, )
                       N is the number of instances
    labels (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in labels is a str
    Returns:
    split_instances_left (numpy.ndarray) = The left side of instances of shape (i + 1, )
    split_instances_right  (numpy.ndarray) = The right side of instances (N - i - 1, )
    split_labels_left (numpy.ndarray) = The left side of labels (i + 1, )
    split_labels_right (numpy.ndarray) = The right side of labels (N - i - 1, )
    """
    split_instances_left = instances[:i, :]
    split_instances_right = instances[i:, :]
    split_labels_left = labels[:i]
    split_labels_right = labels[i:]
    return (
        split_instances_left,
        split_instances_right,
        split_labels_left,
        split_labels_right,
    )


def get_entropy(data, labels):
    """Find the entropy for a dataset
    Args:
    data (numpy.ndarray): Data, numpy array of shape (N, )
                        N is the number of instances
    labels (numpy.ndarray): Class labels, numpy array of shape (N, )
                        Each element in labels is a str
    Returns:
      total_entropy: The total entropy of the dataset
    """
    total_entropy = 0
    for label in np.unique(labels):
        probability = len(data[labels == label]) / len(data)
        total_entropy -= probability * np.log2(probability)

    return total_entropy


def get_information_gain(instances_col, labels):
    """Find the information gain for an attribute
    Args:
    instances_col (numpy.ndarray): Instances Column, numpy array of shape (N, )
                       N is the number of instances
    labels (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in labels is a str
    Returns:
      best_attr_index: The index to the column that we want to split the data on
      best_split_index: The index to to the split point in the column
    """
    total_entropy = get_entropy(instances_col, labels)
    max_info_gain = 0
    split_index = 0
    for index in range(1, len(instances_col)):
        # Split on attribute where attr and class label change
        if (
            labels[index] != labels[index - 1]
            and instances_col[index] != instances_col[index - 1]
        ):
            # Make the split
            split_instances_left = instances_col[:index]
            split_instances_right = instances_col[index:]
            split_labels_left = labels[:index]
            split_labels_right = labels[index:]
            # Get the entropy of each data set
            entropy_left = get_entropy(split_instances_left, split_labels_left)
            entropy_right = get_entropy(split_instances_right, split_labels_right)
            # Get the information gain
            info_gain = (
                total_entropy
                - (len(split_instances_left) / len(instances_col)) * entropy_left
                - (len(split_instances_right) / len(instances_col)) * entropy_right
            )
            # Check if this split leads to the highest information gain
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                split_index = index
    return max_info_gain, split_index


# This function finds the node to split on from a dataset
def find_best_node(instances, labels):
    """Find the best point to create a node in the decision tree
    Args:
    instances (numpy.ndarray): Instances, numpy array of shape (N, K)
                       N is the number of instances
                       K is the number of attributes
    labels (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in labels is a str
    Returns:
      best_attr_index: The index to the column that we want to split the data on
      best_split_index: The index to to the split point in the column
    """
    best_attr_index = 0
    max_info_gain = 0
    best_split_index = 0
    for index in range(0, len(instances[0, :])):
        indices = instances[:, index].argsort()  # returns sorted indices
        sorted_instances = instances[indices]  # sort the instances based off sorted col
        sorted_labels = labels[indices]  # sort the labels based off the sorted col
        # Calculate the info gain and optimal split index for a given attribute
        info_gain, split_index = get_information_gain(
            sorted_instances[:, index], sorted_labels
        )
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_attr_index = index
            best_split_index = split_index

    # Return the best attribute to split on, and the value at which we want to split
    return best_attr_index, best_split_index


def read_dataset(filepath):
    """
    Read .txt file from a specified filepath. (i.e. data/train_full.txt)
    Returns 2 numpy arrays of the instances and the labels
    """
    data = np.loadtxt(filepath, dtype=str, delimiter=",")
    instances = data[:, :-1]
    instances = instances.astype(int)
    labels = data[:, -1]

    return instances, labels


def induce_decision_tree(instances, labels, depth=0):
    """Induce a decision tree for a given dataset

    Args:
    instances (numpy.ndarray): Instances, numpy array of shape (N, K)
                       N is the number of instances
                       K is the number of attributes
    labels (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in labels is a str
    Returns:
      DecisionNode: A decision node object which contains the attribute, value, left node, right node and depth
    """
    # Split this data into tow, left and right
    best_attr_index, best_split_index = find_best_node(instances, labels)

    sorted_indices = instances[:, best_attr_index].argsort()  # returns sorted indices
    sorted_instances = instances[
        sorted_indices
    ]  # sort the instances based off sorted col
    sorted_labels = labels[sorted_indices]  # sort the labels based off the sorted col

    (
        split_instances_left,
        split_instances_right,
        split_labels_left,
        split_labels_right,
    ) = split_dataset(sorted_instances, sorted_labels, best_split_index)

    # If we can no longer split data, return a leaf node
    if (
        len(set(labels)) == 1
        or len(split_instances_left) == 0
        or len(split_instances_right) == 0
    ):
        return Leaf_Node(labels, depth)

    LeftNode = induce_decision_tree(split_instances_left, split_labels_left, depth + 1)
    RightNode = induce_decision_tree(
        split_instances_right, split_labels_right, depth + 1
    )

    # what to return if left node or right node does not exist
    return Decision_Node(
        best_attr_index,
        sorted_instances[best_split_index][best_attr_index],
        LeftNode,
        RightNode,
        depth,
    )


class DecisionTreeClassifier(object):
    """Basic decision tree classifier

    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained

    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self):
        self.is_trained = False
        self.root = None

    def fit(self, x, y):
        """Constructs a decision tree classifier from data

        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K)
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str
        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(
            y
        ), "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        self.root = induce_decision_tree(x, y)

    def predict(self, x):
        """Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K)
                           M is the number of test instances
                           K is the number of attributes

        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """

        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")

        # set up an empty (M, ) numpy array to store the predicted labels
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        for i, instance in enumerate(x):
            predictions[i] = self.root.predict(instance)
        # remember to change this if you rename the variable
        return predictions
