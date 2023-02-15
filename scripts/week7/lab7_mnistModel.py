import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt
import pandas
import sklearn
import pickle
import lab7_mnistLoader
import numpy as np
from numpy import ndarray
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

random_seed = 1
training_data, validation_data, test_data = lab7_mnistLoader.load_data()
# print(training_data[0].shape,training_data[1].shape)

# ndarray.flatten turns a multi-dimension array into a 1-D array
# flatten each
# processed_training_data = [ndarray.flatten(img) for img in training_data[0]]
# processed_validation_data = [ndarray.flatten(img) for img in validation_data[0]]
# processed_test_data = [ndarray.flatten(img) for img in test_data[0]]

training_label = training_data[1]
validation_label = validation_data[1]
test_label = test_data[1]

mlp_model = MLPClassifier(hidden_layer_sizes=10, activation='logistic', solver='sgd', max_iter=10,
                          random_state=random_seed)

mlp_model.fit(training_data[0], training_label)

validation_predictions = mlp_model.predict(validation_data[0])
validation_acc_score = metrics.accuracy_score(validation_label, validation_predictions)
print("Validation Accuracy Score =", validation_acc_score)
disp = metrics.ConfusionMatrixDisplay.from_predictions(validation_label, validation_predictions)
disp.figure_.suptitle("Confusion Matrix")
plt.savefig("validation_cf_matrix.png")
plt.show()

test_predictions = mlp_model.predict(test_data[0])
test_acc_score = metrics.accuracy_score(test_label, test_predictions)
print("Validation Accuracy Score =", validation_acc_score)
disp = metrics.ConfusionMatrixDisplay.from_predictions(test_label, test_predictions)
disp.figure_.suptitle("Confusion Matrix")
plt.savefig("test_cf_matrix.png")
plt.show()
