import numpy
import scipy
import matplotlib
import pandas
import sklearn
import pickle
import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from lab8_mnistLoader import load_data

random_seed = 1

model = pickle.load(open("../../output/bestModel.sav", "rb"))


def check_misclassification(labels, predictions):
    uniques = set(labels)
    # print(uniques)
    instances_dict = {
        "labels": [i for i in uniques],
        "total": [0 for i in uniques]
    }
    # print(instances_dict)
    for i in range(len(labels)):
        if labels[i] != predictions[i]:
            instances_dict["total"][labels[i]] += 1
    return instances_dict


def my_dist(x):
    return np.exp(-x ** 2)

training_data, validation_data, testing_data = load_data()
training_label = training_data[1]
validation_label = validation_data[1]
test_label = testing_data[1]

model.fit(training_data[0], training_label)
prob_dist = model.predict_proba(testing_data[0])
print(prob_dist)
predicts = model.predict(testing_data[0])
number_of_misclassified = check_misclassification(test_label, predicts)
# print(len(test_label))
# print(len(predicts))

# print(number_of_misclassified)
# x = np.arange(number_of_misclassified["labels"][0], number_of_misclassified["labels"][9])
# y = my_dist(x)
# plt.plot(x, y)
# plt.show()

for i in range(1, 6):
    print("k =",i)
    print(top_k_accuracy_score(test_label, prob_dist, k=i))


