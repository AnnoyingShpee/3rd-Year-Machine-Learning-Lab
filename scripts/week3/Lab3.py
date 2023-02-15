from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import pandas

random_seed = 1
data = pandas.read_csv('../../data/wdbc.data', header=None)

# load data, X is an array with all instances with attributes, and y is an array with the target attribute
X, y = load_breast_cancer(return_X_y=True)
# print('Number of instances:', X.shape[0])
# print('Number of attributes:', X.shape[1])

# split data into a train and a test set
# test_size, 70% instances training, 30% instances testing
# random_state, random variates initialised to 1
split_function = ShuffleSplit(n_splits=1, test_size=0.3, random_state=random_seed)
instance_index_list = list(split_function.split(X, y))
train_index_set = instance_index_list[0][0]
test_index_set = instance_index_list[0][1]

# get training/testing instances from the source dataset
X_train = X[train_index_set, :]
y_train = y[train_index_set]
X_test = X[test_index_set, :]
y_test = y[test_index_set]
# print('Training set size:', train_index_set.shape[0])
# print('Testing set size:', test_index_set.shape[0])

#####################################################################
# Bagging classifier
# print('Bagging Classifier')
# number_trees = 500
# dt_training_samples = 200
# bagging_ens = BaggingClassifier(
#     DecisionTreeClassifier(),  # decision tree as base classifier
#     n_estimators=number_trees,  # number of decision trees
#     max_samples=dt_training_samples,  # number of instances used
#     bootstrap=True,  # sampling with replacement
#     random_state=random_seed)
# # train the ensemble
# bagging_ens.fit(X_train, y_train)
#
# # predictions from the same X_train set
# y_predictions_train = bagging_ens.predict(X_train)
# accuracy = accuracy_score(y_train, y_predictions_train)
# print('Accuracy (training set vs predictions):', accuracy)
#
# # predictions from the X_test set
# y_predictions = bagging_ens.predict(X_test)
# # accuracy, y_test vs predictions
# accuracy = accuracy_score(y_test, y_predictions)
# print('Accuracy (testing set vs predictions):', accuracy)
#
# # use the out-of-bag error, as accuracy estimator for unseen instances in the training of base classifier
# bagging_ens = BaggingClassifier(
#     DecisionTreeClassifier(),  # decision tree as base classifier
#     n_estimators=number_trees,  # number of decision trees
#     max_samples=dt_training_samples,  # number of instances used
#     bootstrap=True,  # sampling with replacement
#     oob_score=True,
#     random_state=random_seed)
# # train ensemble
# bagging_ens.fit(X_train, y_train)
# print('Out-Of-Bag (OOB) score', bagging_ens.oob_score_)

print('#################################################')

# number_trees = 100
# dt_training_samples = 100
# bagging_ens_100 = BaggingClassifier(
#     DecisionTreeClassifier(),  # decision tree as base classifier
#     n_estimators=number_trees,  # number of decision trees
#     max_samples=dt_training_samples,  # number of instances used
#     bootstrap=True,  # sampling with replacement
#     random_state=random_seed)
# # train the ensemble
# bagging_ens_100.fit(X_train, y_train)
#
# # predictions from the same X_train set
# y_predictions_train = bagging_ens_100.predict(X_train)
# accuracy = accuracy_score(y_train, y_predictions_train)
# print('Accuracy (training set vs predictions):', accuracy)
#
# # predictions from the X_test set
# y_predictions = bagging_ens_100.predict(X_test)
# # accuracy, y_test vs predictions
# accuracy = accuracy_score(y_test, y_predictions)
# print('Accuracy (testing set vs predictions):', accuracy)
#
# # use the out-of-bag error, as accuracy estimator for unseen instances in the training of base classifier
# bagging_ens_100 = BaggingClassifier(
#     DecisionTreeClassifier(),  # decision tree as base classifier
#     n_estimators=number_trees,  # number of decision trees
#     max_samples=dt_training_samples,  # number of instances used
#     bootstrap=True,  # sampling with replacement
#     oob_score=True,
#     random_state=random_seed)
# # train ensemble
# bagging_ens_100.fit(X_train, y_train)
# print('Out-Of-Bag (OOB) score', bagging_ens_100.oob_score_)

#######################################################################
# Random Forest Classifier
# print('Random Forest Classifier')
# rf_classifier = RandomForestClassifier(
#     random_state=random_seed)
# rf_classifier.fit(X_train, y_train)
# # accuracy on test set
# y_predictions_rf = rf_classifier.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_predictions_rf)
# print('Accuracy (Random Forest):', accuracy_rf)

# number_trees = 10
# rf_classifier_10 = RandomForestClassifier(
#     n_estimators=number_trees,
#     random_state=random_seed)
# rf_classifier_10.fit(X_train, y_train)
# # accuracy on test set
# y_predictions_rf = rf_classifier_10.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_predictions_rf)
# print('Accuracy (Random Forest 10):', accuracy_rf)  # 0.935672514619883
#
# number_trees = 50
# rf_classifier_50 = RandomForestClassifier(
#     n_estimators=number_trees,
#     random_state=random_seed)
# rf_classifier_50.fit(X_train, y_train)
# # accuracy on test set
# y_predictions_rf = rf_classifier_50.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_predictions_rf)
# print('Accuracy (Random Forest 50):', accuracy_rf)  # 0.9415204678362573
#
# number_trees = 100
# rf_classifier_100 = RandomForestClassifier(
#     n_estimators=number_trees,
#     random_state=random_seed)
# rf_classifier_100.fit(X_train, y_train)
# # accuracy on test set
# y_predictions_rf = rf_classifier_100.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_predictions_rf)
# print('Accuracy (Random Forest 100):', accuracy_rf)  # 0.9473684210526315
#
# number_trees = 200
# rf_classifier_200 = RandomForestClassifier(
#     n_estimators=number_trees,
#     random_state=random_seed)
# rf_classifier_200.fit(X_train, y_train)
# # accuracy on test set
# y_predictions_rf = rf_classifier_200.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_predictions_rf)
# print('Accuracy (Random Forest 200):', accuracy_rf)  # 0.9532163742690059
#
# number_trees = 500
# rf_classifier_500 = RandomForestClassifier(
#     n_estimators=number_trees,
#     random_state=random_seed)
# rf_classifier_500.fit(X_train, y_train)
# # accuracy on test set
# y_predictions_rf = rf_classifier_500.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_predictions_rf)
# print('Accuracy (Random Forest 500):', accuracy_rf)  # 0.9473684210526315
#
# number_trees = 1000
# rf_classifier_1000 = RandomForestClassifier(
#     n_estimators=number_trees,
#     random_state=random_seed)
# rf_classifier_1000.fit(X_train, y_train)
# # accuracy on test set
# y_predictions_rf = rf_classifier_1000.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_predictions_rf)
# print('Accuracy (Random Forest 1000):', accuracy_rf)  # 0.9473684210526315

print('#################################################')

#############################################################
# Grid Search Classifier
# print('Grid Search on Random Forest')
#
number_trees = 200
rf_classifier = RandomForestClassifier(
    n_estimators=number_trees,
    random_state=random_seed)
rf_classifier.fit(X_train, y_train)

gs_eval_rf = GridSearchCV(
    estimator=rf_classifier,
    param_grid={'max_features': [5, 10],
                'max_leaf_nodes': [20, 30, 40]})

gs_eval_rf.fit(X_train, y_train)

print('Random Forest best parameters :', gs_eval_rf.best_params_)
print('Random Forest best score :', gs_eval_rf.best_score_)

print('#################################################')

#################################################
# Gradient Boosting Classifier
print('Gradient Boosting Classifier')

gb_classifier = GradientBoostingClassifier(random_state=random_seed)
gb_classifier.fit(X_train, y_train)
y_predictions_gb = gb_classifier.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_predictions_gb)
print('Accuracy (Gradient Boosting):', accuracy_gb)

print('#################################################')

print()
gs_eval_gb = GridSearchCV(
    estimator=gb_classifier
)


