import numpy as np

def cross_entropy(actual, predicted):
    return -np.sum(actual * np.log(predicted))

# One Hot Encode:
# Class 0 = np.array([1, 0, 0])
# Class 1 = np.array([0, 1, 0])
# Class 2 = np.array([0, 0, 1])
Y = np.array([1, 0, 0])

Y_predicted = np.array([0.6, 0.3, 0.1])
X_predicted = np.array([0.1, 0.3, 0.6])

Bad_Output = cross_entropy(Y, X_predicted)
Good_Output = cross_entropy(Y, Y_predicted)

print(f"Good: {Good_Output:.4f}")
print(f"Bad: {Bad_Output:.4f}")