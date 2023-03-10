# load packages
# dataset
from sklearn.datasets import load_breast_cancer
# ML algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# KFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

# plot cross-validation mean accuracy
import matplotlib.pyplot as plt

random_seed = 1  # random number generator seed
kfold_splits = 15  # number of folds
# https://scikit-learn.org/stable/modules/model_evaluation.html
# scoring (Classification) = [  "accuracy", "balanced_accuracy", "top_k_accuracy", "average_precision",
#                               "neg_brier_score", "f1", "f1_micro", "f1_macro", "f1_weighted", "f1_samples",
#                               "neg_log_loss", "precision", "recall", "jaccard", "roc_auc", "roc_auc_ovr",
#                               "roc_auc_ovo", "roc_auc_ovr_weighted", "roc_auc_ovo_weighted"]
scoring_options = 'balanced_accuracy'
# load ALL data. The cross-validation will split ALL data into train
# and test sets for each fold
X, y = load_breast_cancer(return_X_y=True)

# create ML algorithms with default values
rf_classifier = RandomForestClassifier()
knn_classifier = KNeighborsClassifier()
svm_classifier = SVC()

# create k-fold validator objects
rf_kfold = KFold(n_splits=kfold_splits, random_state=random_seed, shuffle=True)
knn_kfold = KFold(n_splits=kfold_splits, random_state=random_seed, shuffle=True)
svm_kfold = KFold(n_splits=kfold_splits, random_state=random_seed, shuffle=True)

# cross-validation report
rf_cross_val = cross_val_score(rf_classifier, X, y, cv=rf_kfold, scoring=scoring_options, verbose=True)
print(rf_cross_val)
print("rf_cross_val mean", scoring_options, rf_cross_val.mean())
print("rf_cross_val std dev", scoring_options, rf_cross_val.std())

knn_cross_val = cross_val_score(knn_classifier, X, y, cv=knn_kfold, scoring=scoring_options, verbose=True)
print(knn_cross_val)
print("knn_cross_val mean", scoring_options, knn_cross_val.mean())
print("knn_cross_val std dev", scoring_options, knn_cross_val.std())

svm_cross_val = cross_val_score(svm_classifier, X, y, cv=svm_kfold, scoring=scoring_options, verbose=True)
print(svm_cross_val)
print("svm_cross_val mean", scoring_options, svm_cross_val.mean())
print("svm_cross_val std dev", scoring_options, svm_cross_val.std())


# visualise the mean and std dev of accuracy
cv_summary = [rf_cross_val, knn_cross_val, svm_cross_val]
labels = ["Random Forest", "KNN", "SVM"]
fig = plt.figure()
fig.suptitle("ML Algorithm {} Comparison with Cross-Validation : {} folds".format(scoring_options, kfold_splits))
ax = fig.add_subplot(111)
plt.boxplot(cv_summary, showmeans=True, showfliers=False)
plt.ylim(0.8, 1.0)
ax.set_ylabel(scoring_options)
ax.set_xticklabels(labels)
plt.show()
# Green mark = Mean Accuracy
# Orange line = Median Accuracy
