import numpy
import scipy
import matplotlib
import pandas
import sklearn
from math import sqrt
from sklearn.model_selection import train_test_split

# read data using pandas
data = pandas.read_csv('../../data/knn_example_data.csv', header=None, index_col=None)
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# separate data into X, y structures to follow scikit-learn conventions
# X: features of the input data
# y: target/label/class vector of the input data
X = data[[2, 3]].values  # columns 2 and 3 are the instance attributes
y = data[1].values  # column 1 is the target/class

# split data 70/30
random_seed = 1
test_set_size = 0.3

# the following command returns you data as two separate sets, train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=random_seed)
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


def confusion_matrix(actual, prediction):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(actual)):
        if actual[i] == 1 and prediction[i] == 1:
            true_positive += 1
        elif actual[i] == 0 and prediction[i] == 0:
            true_negative += 1
        elif actual[i] == 0 and prediction[i] == 1:
            false_positive += 1
        elif actual[i] == 1 and prediction[i] == 0:
            false_negative += 1

    return true_positive, true_negative, false_positive, false_negative


class NNClassifier:
    def __init__(self, train_attributes, test_attributes, train_values, test_values):
        self.train_attributes = train_attributes
        self.test_attributes = test_attributes
        self.train_values = train_values
        self.test_values = test_values

    def get_euclidean_distance(self, train, query):
        """
            Calculate Euclidean Distance function
            :param train: train set
            :param query: query vector
            :return: square root of square distance
        """
        sqr_diff = 0
        for i in range(len(train)):
            sqr_diff += (query[i] - train[i]) ** 2

        return sqrt(sqr_diff)

    def predict(self, query):
        """
        1-NN Predict function
        :param self: use train_attributes and train_values
        :param query: test case
        :return: class label of the closest instance to the test case
        """
        arr = []
        for i in range(len(self.train_attributes)):
            arr.append(self.get_euclidean_distance(self.train_attributes[i], query))

        predict_class = self.train_values[arr.index(min(arr))]

        return predict_class

    def test_data(self):
        predicted_list = []
        for index in range(len(X_test)):
            predicted_list.append(self.predict(X_test[index]))

        return predicted_list


# def get_euclidean_distance(train, query):
#     """
#         Calculate Euclidean Distance function
#         :param train: train set
#         :param query: query vector
#         :return: square root of square distance
#     """
#     sqr_diff = 0
#     for i in range(len(train)):
#         sqr_diff += (query[i] - train[i]) ** 2
#
#     return sqrt(sqr_diff)
#
#
# def predict(x_set, y_set, query):
#     """
#     1-NN Predict function
#     :param x_set: train set
#     :param y_set: class vector
#     :param query: test case
#     :return: class label of the closest instance to the test case
#     """
#     arr = []
#     for i in range(len(x_set)):
#         arr.append(get_euclidean_distance(x_set[i], query))
#
#     # Get smallest value (Euclidean distance) of arr, get index of smallest value, get value of y_set using said index
#     predict_class = y_set[arr.index(min(arr))]
#
#     return predict_class
#
# predicted_list = []
# for index in range(len(X_test)):
#     predicted_list.append(predict(X_train, y_train, X_test[index]))
#
# print(y_test)
# print(predicted_list)
# print(confusion_matrix(y_test, predicted_list))

nn_classifier = NNClassifier(X_train, X_test, y_train, y_test)
predictions = nn_classifier.test_data()
print(y_test)
print(predictions)
print(confusion_matrix(y_test, predictions))



