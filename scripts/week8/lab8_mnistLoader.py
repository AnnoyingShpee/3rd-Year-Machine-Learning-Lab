import numpy
import scipy
import matplotlib
import pandas
import sklearn
import pickle


def load_data():
    """Return the MNIST data as a tuple containing the training data,
     the validation data, and the test data.

     The ``training_data`` is returned as a tuple with two entries.
     The first entry contains the actual training images. This is a
     numpy ndarray with 50,000 entries. Each entry is, in turn, a
     numpy ndarray with 784 values, representing the 28 * 28 = 784
     pixels in a single MNIST image.

     The second entry in the ``training_data`` tuple is a numpy ndarray
     containing 50,000 entries. Those entries are just the digit
     values (0...9) for the corresponding images contained in the first
     entry of the tuple.

     The ``validation_data`` and ``test_data`` are similar, except
     each contains only 10,000 images.

     """

    # The filename of the dataset in relative path, modifying it as necesseary
    filename = "../../data/mnist.pkl"

    with open(filename, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='iso-8859-1')
        f.close()
    return training_data, validation_data, test_data

