import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice
from classification import induce_decision_tree, read_dataset

def precision(y_gold, y_prediction):
    """ Compute the precision score per class given the ground truth and predictions
        
    Also return the macro-averaged precision across classes.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (precisions, macro_precision) where
            - precisions is a np.ndarray of shape (C,), where each element is the 
              precision for class c
            - macro-precision is macro-averaged precision (a float) 
    """

    confusion = confusion_matrix(y_gold, y_prediction)
    p = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[:, c]) > 0:
            p[c] = confusion[c, c] / np.sum(confusion[:, c])

    ## Alternative solution without computing the confusion matrix
    #class_labels = np.unique(np.concatenate((y_gold, y_prediction)))
    #p = np.zeros((len(class_labels), ))
    #for (c, label) in enumerate(class_labels):
    #    indices = (y_prediction == label) # get instances predicted as label
    #    correct = np.sum(y_gold[indices] == y_prediction[indices]) # intersection
    #    if np.sum(indices) > 0:
    #        p[c] = correct / np.sum(indices)     

    # Compute the macro-averaged precision
    macro_p = 0.
    if len(p) > 0:
        macro_p = np.mean(p)
    
    return (p, macro_p)


def recall(y_gold, y_prediction):
    """ Compute the recall score per class given the ground truth and predictions
        
    Also return the macro-averaged recall across classes.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (recalls, macro_recall) where
            - recalls is a np.ndarray of shape (C,), where each element is the 
                recall for class c
            - macro-recall is macro-averaged recall (a float) 
    """

    confusion = confusion_matrix(y_gold, y_prediction)
    r = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[c, :]) > 0:
            r[c] = confusion[c, c] / np.sum(confusion[c, :])

    ## Alternative solution without computing the confusion matrix
    #class_labels = np.unique(np.concatenate((y_gold, y_prediction)))
    #r = np.zeros((len(class_labels), ))
    #for (c, label) in enumerate(class_labels):
    #    indices = (y_gold == label) # get instances for current label
    #    correct = np.sum(y_gold[indices] == y_prediction[indices]) # intersection
    #    if np.sum(indices) > 0:
    #        r[c] = correct / np.sum(indices)     

    # Compute the macro-averaged recall
    macro_r = 0.
    if len(r) > 0:
        macro_r = np.mean(r)
    
    return (r, macro_r)


def f1_score(y_gold, y_prediction):
    """ Compute the F1-score per class given the ground truth and predictions
        
    Also return the macro-averaged F1-score across classes.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where
            - f1s is a np.ndarray of shape (C,), where each element is the 
              f1-score for class c
            - macro-f1 is macro-averaged f1-score (a float) 
    """

    (precisions, macro_p) = precision(y_gold, y_prediction)
    (recalls, macro_r) = recall(y_gold, y_prediction)

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions), ))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    # Compute the macro-averaged F1
    macro_f = 0.
    if len(f) > 0:
        macro_f = np.mean(f)
    
    return (f, macro_f)


def accuracy(y_gold, y_prediction):
    """ Compute the accuracy given the ground truth and predictions

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        float : the accuracy
    """

    assert len(y_gold) == len(y_prediction)  
    
    try:
        return np.sum(y_gold == y_prediction) / len(y_gold)
    except ZeroDivisionError:
        return 0

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

def confusion_matrix(y_gold, y_prediction, class_labels=None):
    """ Compute the confusion matrix.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        class_labels (np.ndarray): a list of unique class labels. 
                               Defaults to the union of y_gold and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes. 
                   Rows are ground truth per class, columns are predictions
    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    # for each correct class (row), 
    # compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
        indices = (y_gold == label)
        gold = y_gold[indices]
        predictions = y_prediction[indices]

        # quick way to get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion

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
    num_trees = 70
    num_attributes = 5
    trees = []

    # Train the trees
    for i in range(num_trees):
        trees.append(induce_random_decision_tree(x_train, y_train, num_attributes))
        # print(trees[i])
            

    # Predict on trees
    predictions = []
    for i, instance in enumerate(x_test):
        answers = []
        for j, tree in enumerate(trees):
            # print(f"{instance} -> {tree.predict(instance)}")
            answers.append(tree.predict(instance))
        
        new_answers = np.array(answers)
        vals, counts = np.unique(new_answers, return_counts=True)
        index = np.argmax(counts)
        predictions.append(vals[index])

    
    return np.array(predictions)
    

train_instances, train_labels = read_dataset("data/train_full.txt")
test_instances, test_labels = read_dataset("data/test.txt")
predictions = train_and_predict(train_instances, train_labels, test_instances, 0, 0)
print(confusion_matrix(test_labels, predictions))
print(f"Accuracy: {accuracy(test_labels, predictions)}")
print(f"Precision: {precision(test_labels, predictions)}")
print(f"Recall: {recall(test_labels, predictions)}")
print(f"F1: {f1_score(test_labels, predictions)}")