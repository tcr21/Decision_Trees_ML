##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function.
#             You are free to add any other methods as needed.
##############################################################################

from classification import induce_decision_tree
import numpy as np
from numpy.random import choice


class Filter_Node:
    def __init__(self, attributes, node):
        self.attributes = attributes
        self.next = node

    def __repr__(self):
        tab = "\t"
        return f"Filter Node attributes: {self.attributes}"

    def predict(self, instance):
        instance = instance[self.attributes]
        return self.next.predict(instance)

def induce_random_decision_tree(instances, labels, num_attributes, depth=0):
    # Randomly sample instances
    indices_observations = choice(len(instances), size=(len(instances)))
    instances_copy = np.array(instances[indices_observations])
    labels_copy = np.array(labels[indices_observations])
    
    # Randomly select attributes
    indices_attributes = choice(len(instances[0]), size=num_attributes)
    instances_copy = np.array(instances_copy[:, indices_attributes])

    # Induce a normal decision tree from that
    return Filter_Node(indices_attributes, induce_decision_tree(instances_copy, labels_copy))

def train_and_predict(x_train, y_train, x_test, x_val, y_val):
    """Interface to train and test the new/improved decision tree.

    This function is an interface for training and testing the new/improved
    decision tree classifier.

    x_train and y_train should be used to train your classifier, while
    x_test should be used to test your classifier.
    x_val and y_val may optionally be used as the validation dataset.
    You can just ignore x_val and y_val if you do not need a validation dataset.

    Args:
    x_train (numpy.ndarray): Training instances, numpy array of shape (N, K)
                       N is the number of instances
                       K is the number of attributes
    y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in y is a str
    x_test (numpy.ndarray): Test instances, numpy array of shape (M, K)
                            M is the number of test instances
                            K is the number of attributes
    x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K)
                       L is the number of validation instances
                       K is the number of attributes
    y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )
    """

    #######################################################################
    #                 ** TASK 4.1: COMPLETE THIS FUNCTION **
    #######################################################################
    # Validation performed separately, optimal value hard coded
    num_trees = 70 
    num_attributes = 5
    trees = []

    # Train the trees
    for i in range(num_trees):
        trees.append(induce_random_decision_tree(x_train, y_train, num_attributes))
            
    # Predict on trees
    predictions = []
    for i, instance in enumerate(x_test):
        answers = []
        for j, tree in enumerate(trees):
            answers.append(tree.predict(instance))
        
        new_answers = np.array(answers)
        vals, counts = np.unique(new_answers, return_counts=True)
        index = np.argmax(counts)
        predictions.append(vals[index])

    return np.array(predictions)
