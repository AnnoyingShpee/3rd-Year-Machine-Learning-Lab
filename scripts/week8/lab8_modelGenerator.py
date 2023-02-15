import numpy
import scipy
import matplotlib
import pandas
import sklearn
import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from lab8_mnistLoader import load_data

random_seed = 1

training_data, validation_data, testing_data = load_data()
training_label = training_data[1]
validation_label = validation_data[1]
test_label = testing_data[1]

# print("Training: ")
# print(training_data)
# print("Validation: ")
# print(validation_data)
# print("Testing: ")
# print(testing_data)

# mlp_model = MLPClassifier(activation='logistic', solver='sgd', random_state=random_seed)

# param_grid = dict(alpha=[0.001, 0.01, 0.1], hidden_layer_sizes=[10, 20, 30], max_iter=[10, 20, 30])
# grid = GridSearchCV(estimator=mlp_model, param_grid=param_grid)
# grid_result = grid.fit(training_data[0], training_label)
#
# print('Grid Search best parameters :', grid.best_params_)
# print('Grid Search best score :', grid.best_score_)

best_model = MLPClassifier(hidden_layer_sizes=40, activation='logistic', solver='sgd', max_iter=30,
                           random_state=random_seed, alpha=0.001, learning_rate_init=0.3)

# best_model.fit(training_data[0], training_label)
# predictions = best_model.predict(testing_data[0])
# acc_score = accuracy_score(test_label, predictions)
#
# print(acc_score)
pickle.dump(best_model, open("../../output/bestModel.sav", "wb"))
