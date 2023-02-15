import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt
import pandas
import sklearn

test_target = pandas.read_csv('../../data/test_target.csv', header=None, index_col=False, sep=',')
test_target_array = test_target[0].values
predictions = pandas.read_csv('../../data/predictions.csv', header=None, index_col=False, sep=',')
predictions_array = predictions[0].values
train_data = pandas.read_csv('../../data/train_data.csv', header=None, index_col=False, sep=',')
test_data = pandas.read_csv('../../data/test_data.csv', header=None, index_col=False, sep=',')


def confusion_matrix(actual, predict):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(actual)):
        if actual[i] == 1 and predict[i] == 1:
            true_positive += 1
        elif actual[i] == 0 and predict[i] == 0:
            true_negative += 1
        elif actual[i] == 0 and predict[i] == 1:
            false_positive += 1
        elif actual[i] == 1 and predict[i] == 0:
            false_negative += 1

    return true_positive, true_negative, false_positive, false_negative


def calc_accuracy(true_positive, true_negative, false_positive, false_negative):
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)


def calc_tpr_sensitivity(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)


def calc_tnr_specificity(true_negative, false_positive):
    return true_negative / (true_negative + false_positive)


def calc_balanced_accuracy(sensitivity, specificity):
    return (tpr + tnr) / 2


def calc_precision(true_positive, false_positive):
    return true_positive / (true_positive + false_positive)


def calc_f1(precision_var, recall):
    return 2 * ((precision_var * recall) / (precision_var + recall))


tp, tn, fp, fn = confusion_matrix(test_target_array, predictions_array)
cm = confusion_matrix(test_target_array, predictions_array)
print('TP =', tp)
print('TN =', tn)
print('FP =', fp)
print('FNS =', fn)
print('CM =', cm)

accuracy = calc_accuracy(tp, tn, fp, fn)
tpr = calc_tpr_sensitivity(tp, fn)
tnr = calc_tnr_specificity(tn, fp)
balanced_accuracy = calc_balanced_accuracy(tpr, tnr)
precision = calc_precision(tp, fp)
f1 = calc_f1(precision, tpr)

print('Accuracy =', accuracy)
print('TPR/Sensitivity/Recall =', tpr)
print('TNR/Specificity =', tnr)
print('Balanced Accuracy =', balanced_accuracy)
print('Precision =', precision)
print('F1 Score =', f1)
