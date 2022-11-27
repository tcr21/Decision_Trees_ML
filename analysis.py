import numpy as np
import matplotlib.pyplot as plt

def read_dataset(filepath):
    """
    Read .txt file from a specified filepath. (i.e. data/train_full.txt)
    Returns 2 numpy arrays of the instances and the labels
    """
    data = np.loadtxt(filepath, dtype=np.str, delimiter=",")
    instances = data[:,:-1]
    instances = instances.astype(np.int)
    labels = data[:,-1]

    return instances, labels


instances, labels = read_dataset("data/toy.txt")

def get_entropy(data, labels):
    H_total = 0
    for label in np.unique(labels):
        probability = len(data[labels==label]) / len(data)
        H_total -= probability * np.log2(probability)
    
    return H_total

# Get total entropy
H_total = get_entropy(instances, labels)
print("Total entropy: ", H_total)

# For each attribute
    # Sort
    # Loop through attribute
        # If classes differ,
            # Select as split point
            # Calculate new information gain
            # If this is better than before, store attribute, split point, IG
            
# for attribute_index in range(len(instances[0])):
#     sorted_indices = np.argsort(instances, attribute_index)

indices = instances[:,0].argsort() # returns sorted indices
instances = instances[indices]
labels = labels[indices]

max_IG = 0
best_split = 0
for index in range(1, len(indices)):
    if labels[index] != labels[index-1]:
        set1 = instances[:index]
        set2 = instances[index:]
        print(f"Split at {index}")
        H1 = get_entropy(set1, labels[:index])
        H2 = get_entropy(set2, labels[index:])
        IG = H_total - H1 - H2
        print(f"H1: {H1}, H2: {H2}, IG: {IG}")
        if IG > max_IG: 
            max_IG = IG
            best_split = index

print(f"Split at {best_split} for IG of {max_IG}")

# split_set1 = data[data[:,0] > threshold, :]
# split_set2 = data[data[:,0] <= threshold, :]
