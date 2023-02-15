import math
import numpy
import scipy
import matplotlib
import pandas
import sklearn


def calc_entropy(frac_1, frac_2):
    entropy = -((frac_1)*(math.log2(frac_1)) + (frac_2)*(math.log2(frac_2)))
    return entropy




print(calc_entropy((1/5), (4/5)))
print(calc_entropy((3/5), (2/5)))
print(calc_entropy((1/3), (2/3)))
