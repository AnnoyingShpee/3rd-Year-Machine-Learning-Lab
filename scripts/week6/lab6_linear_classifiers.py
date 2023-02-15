import numpy as np
import scipy
import matplotlib
import pandas
import sklearn

# load data
from matplotlib import pyplot as plt

from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

rand_seed = 1
# create synthetic data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                           n_clusters_per_class=1, class_sep=2, random_state=rand_seed)

# print data size
print(X.shape)
print(y.shape)

#####################################################
# Linear Discriminant Analysis
# plot train data
plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
ax.set_xlabel('X_1')
ax.set_ylabel('X_2')
plt.xlim(min(X[:, 0]), max(X[:, 0]))
plt.ylim(min(X[:, 1]), max(X[:, 1]))
plt.savefig(f"../../output/linear_classifier_synthetic_data.png")
plt.show()

lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(X, y)
lda_y_predicts = lda_classifier.predict(X)
lda_accuracy = accuracy_score(y, lda_y_predicts)
print(f"LDA Coefficients:{lda_classifier.coef_.shape}")
print(f"LDA accuracy:{lda_accuracy}")

# plot predictions
plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=lda_y_predicts)
ax.set_xlabel('X_1')
ax.set_ylabel('X_2')
plt.xlim(min(X[:, 0]), max(X[:, 0]))
plt.ylim(min(X[:, 1]), max(X[:, 1]))
plt.savefig(f"../../output/linear_discriminant_analysis_synthetic_data.png")
plt.show()
#####################################################

#####################################################
# Linear Support Vector
# create SVM classifier
svm_classifier = LinearSVC()
svm_classifier.fit(X, y)
svm_y_predicts = svm_classifier.predict(X)
svm_accuracy = accuracy_score(y, svm_y_predicts)
print(f"SVM Coefficients:{svm_classifier.coef_.shape}")
print(f"SVM accuracy:{svm_accuracy}")

# plot predictions
plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=svm_y_predicts)
ax.set_xlabel('X_1')
ax.set_ylabel('X_2')
plt.xlim(min(X[:, 0]), max(X[:, 0]))
plt.ylim(min(X[:, 1]), max(X[:, 1]))
plt.savefig(f"../../output/linear_support_vector_synthetic_data.png")
plt.show()