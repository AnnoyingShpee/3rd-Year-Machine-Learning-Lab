import numpy as np
import scipy
import matplotlib
import pandas
import sklearn

# load data
from matplotlib import pyplot as plt
from sklearn.linear_model import Perceptron

data = pandas.read_csv('../../data/perceptron_data.csv', index_col=None, header=None)

# slice data to get X and y
X = data[[0, 1]].values  # columns 0, 1 are attributes x1, x2
y = data[2].values  # column 2 is the target

# separate data per class, scatter plot
# slide data X to get odd rows (class -1)
X_class_negative = X[::2, :] # Same as [start_index : stop_index : iteration_length] = [0 : -1 : 2]
X_class_positive = X[1::2, :]  # slide data X to get even rows (class +1)

# init variables
learning_rate = 1
max_iter = 5
w = [-1, 2]


# YOUR CODE HERE
def perceptron_train(weight, instance):
    result = 0
    for i in range(len(weight)):
        result += (weight[i] * instance[i])
    if result >= 0:
        return 1
    else:
        return -1


# outer loop through all data
epoch = 0
for epoch in range(max_iter):
    print("----------------------------------------")
    print(f"epoch {epoch+1}")
    w_updated = False
    i = 0
    for i in range(len(X)):
        # print instance_ID and actual weights
        print(f"{i+1}\tw={w}")
        # predict y for instance X_i
        y_i_predict = perceptron_train(w, X[i])
        # print actual data
        print(f"\tx={X[i]}\tclass:{y[i]}\tpredict:{y_i_predict}")

        # if actual class is different to prediction
        if y[i] != y_i_predict:
            w_updated = True  # switch flag

        # adjust weights
        # delta_w = 0.5 * eta * (actual - predict) * instance
        # X[i] = array
        delta_w = 0.5 * learning_rate * (y[i] - y_i_predict) * X[i]
        w = w + delta_w
        # all(array) returns a boolean whether all items in iterable object are True.
        # any(array) is the same except whether one of the items is True
        if not all(delta_w):
            print(f"\tcorrect, w:{w}")
        else:
            print(f"\twrong, w:{w}")


    print("----------------------------------------")
    print(f"epochs:{epoch}")
    print(f"w:{w}")
    print("----------------------------------------")
    print(f"plot data and decision boundary")
    # plot data with the decision boundary
    # calculate the decision boundary
    y_min, y_max = -5, 5
    xx = np.linspace(y_min, y_max, 2)  # create only two points at -6, 6
    m = -w[0] / w[1]
    c = -1 / w[1]
    yy = m * xx - c
    print(xx)

    # plot data and decision boundary
    plt.figure(figsize=(5, 5))
    ax = plt.axes()

    # add data points
    plt.scatter(X_class_negative[:, 0], X_class_negative[:, 1], c="blue")  # class -1
    plt.scatter(X_class_positive[:, 0], X_class_positive[:, 1], c="red")  # class +1

    # add decision boundary
    plt.plot(xx, yy, c="green")

    # format plot
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    plt.xlim(y_min, y_max)  # is squared, so y_min, y_max
    plt.ylim(y_min, y_max)  # is squared, so y_min, y_max
    plt.title(f"The Perceptron, epoch:{epoch + 1}/{max_iter}")
    plt.show()
    # plot data with the decision boundary, end

    # if at the end of the full pass w_updated remains False, then
    # weights didn't change, so all instances in the data were
    # classified correctly
    if not w_updated:
        print("Perceptron has converged!")
        break

    # increment epoch counter by 1
    # epoch += 1

##########################################################
# Using built-in Perceptron Classifier
max_iter = 100
shuffle = False
verbose = True

perceptron_classifier = Perceptron(max_iter=max_iter, shuffle=shuffle, verbose=verbose)
perceptron_classifier.fit(X, y)
y_min, y_max = -5, 5
print(y_min)
xx = np.linspace(y_min, y_max, 2)  # create only two points at -6, 6
w = perceptron_classifier.coef_[0]
m = -w[0] / w[1]
c = perceptron_classifier.intercept_[0] / w[1]
yy = m * xx - c

# plot data and decision boundary
plt.figure(figsize=(5, 5))
ax = plt.axes()

# add data points
plt.scatter(X_class_negative[:, 0], X_class_negative[:, 1], c="blue")  # class -1
plt.scatter(X_class_positive[:, 0], X_class_positive[:, 1], c="red")  # class +1

# add decision boundary
plt.plot(xx, yy, c="green")

# format plot
ax.set_xlabel('X_1')
ax.set_ylabel('X_2')
plt.xlim(y_min, y_max)  # is squared, so y_min, y_max
plt.ylim(y_min, y_max)  # is squared, so y_min, y_max
plt.title(f"The Perceptron, epoch:")
plt.show()
# plot data with the decision boundary, end
