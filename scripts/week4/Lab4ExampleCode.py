# load packages
# data manipulation and plot
import pandas as pd
import matplotlib.pyplot as plt

# confusion-matrix functions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# read a csv file and return the first column
def read_csv(filename):
    df = pd.read_csv(filename, header=None, index_col=False, sep=',')
    return df[0].values


# read data, actual values and predictions
y_test = read_csv('../../data/test_target.csv')
y_predict = read_csv('../../data/predictions.csv')

# compute confusion matrix and get the counts
tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
specificity = tn / (tn+fp)
print('TP', tp)
print('FN', fn)
print('FP', fp)
print('TN', tn)
print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_predict))
print('Specificity:', specificity)
print('Report\n', classification_report(y_test, y_predict))

# visualise the confusion matrix
cm = confusion_matrix(y_test, y_predict, labels=[1, 0])
print('cm\n', cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
disp.plot()
plt.show()
