import numpy as np
import scipy
import matplotlib
import pandas
import sklearn
import pickle
import lab9_mnistLoader as mnl

default_filter = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])

lab_filter = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ])

horizontal_filter = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ])

vertical_filter = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])

class Filter:
    def __init__(self, kernel=lab_filter, padding=0, stride=1):
        self.kernel = kernel
        self.padding = padding
        self.stride = stride

    def feature(self, input_matrix):
        kernel_x_size = self.kernel.shape[0]
        kernel_y_size = self.kernel.shape[1]
        input_x_size = input_matrix.shape[0]
        input_y_size = input_matrix.shape[1]
        output_x_size = int(((input_x_size - kernel_x_size + 2 * self.padding) / self.stride) + 1)
        output_y_size = int(((input_y_size - kernel_y_size + 2 * self.padding) / self.stride) + 1)
        output_matrix = np.zeros((output_x_size, output_y_size))
        padded_matrix = None
        if self.padding != 0:
            padded_matrix = np.zeros((input_x_size+self.padding*2, input_y_size+self.padding*2))
            padded_matrix[int(self.padding):int(-1 * self.padding),
                          int(self.padding):int(-1 * self.padding)] = input_matrix
        else:
            padded_matrix = input_matrix
        # Iterate through y dimension elements
        for y in range(input_matrix.shape[1]):
            # Checks if at end of input in y direction. Exits if true
            if y > input_matrix.shape[1] - kernel_y_size:
                break
            # Makes sure step size is the same as stride. Only move if y has moved by the specified Strides
            if y % self.stride == 0:
                for x in range(input_matrix.shape[0]):
                    if x > input_matrix.shape[0] - kernel_x_size:
                        break
                    try:
                        # Only move if x has moved by the specified Strides
                        if x % self.stride == 0:
                            r = x // self.stride
                            c = y // self.stride
                            output_matrix[r, c] = (
                                    self.kernel *
                                    padded_matrix[x: x + kernel_x_size, y: y + kernel_y_size]).sum()

                    except Exception as e:
                        print(e)
                        break
        return output_matrix


class Pooling:
    def __init__(self, pool_window=(2, 2), padding=0, stride=1):
        self.pooling_window = np.zeros(pool_window)
        self.padding = padding
        self.stride = stride

    def max_pooling(self, input_matrix):
        pool_x_size = self.pooling_window.shape[0]
        pool_y_size = self.pooling_window.shape[1]
        input_x_size = input_matrix.shape[0]
        input_y_size = input_matrix.shape[1]
        output_x_size = int(((input_x_size - pool_x_size + 2 * self.padding) / self.stride) + 1)
        output_y_size = int(((input_y_size - pool_y_size + 2 * self.padding) / self.stride) + 1)
        output_matrix = np.zeros((output_x_size, output_y_size))
        padded_matrix = None
        if self.padding != 0:
            padded_matrix = np.zeros((input_x_size + self.padding * 2, input_y_size + self.padding * 2))
            padded_matrix[int(self.padding):int(-1 * self.padding),
                          int(self.padding):int(-1 * self.padding)] = input_matrix
        else:
            padded_matrix = input_matrix
        # Iterate through y dimension elements
        for y in range(input_matrix.shape[1]):
            # Checks if at end of input in y direction. Exits if true
            if y > input_matrix.shape[1] - pool_y_size:
                break
            # Makes sure step size is the same as stride. Only move if y has moved by the specified Strides
            if y % self.stride == 0:
                for x in range(input_matrix.shape[0]):
                    if x > input_matrix.shape[0] - pool_x_size:
                        break
                    try:
                        # Only move if x has moved by the specified Strides
                        if x % self.stride == 0:
                            r = x // self.stride
                            c = y // self.stride
                            output_matrix[r, c] = padded_matrix[x: x + pool_x_size, y: y + pool_y_size].max()
                    except Exception as e:
                        print(e)
                        break
        return output_matrix


if __name__ == "__main__":
    random_seed = 1
    training_data, validation_data, test_data = mnl.load_data()
    training_label = training_data[1]
    validation_label = validation_data[1]
    test_label = test_data[1]

    # Part B: 1 to 3
    # filter_class = Filter()
    # pooling_class = Pooling()
    # image = np.array([
    #     [1, 1, 1, 0, 1],
    #     [1, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 0, 1, 1]
    # ])
    # feature = filter_class.feature(image)
    # print(feature)
    # pooled_matrix = pooling_class.max_pooling(feature)
    # print(pooled_matrix)

    # Part B: 4 to 6
    filter_class = Filter(kernel=lab_filter)
    pooling_class = Pooling()
    for i in range(10):
        # Reshape array to matrix
        image = training_data[0][i].reshape(28, 28)
        feature = filter_class.feature(image)
        print(feature)
        pooled_matrix = pooling_class.max_pooling(feature)
        print(pooled_matrix)

