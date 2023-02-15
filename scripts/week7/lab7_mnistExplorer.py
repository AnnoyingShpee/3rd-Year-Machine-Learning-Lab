import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt
import pandas
import sklearn
import pickle
import lab7_mnistLoader

training_data, validation_data, test_data = lab7_mnistLoader.load_data()


# print(len(training_data))  # 2
# print(len(validation_data))
# print(len(test_data))

# print(training_data)
# print(validation_data)
# print(test_data)

# print(len(training_data[0]))
# print(training_data[0])
# print(len(validation_data[0]))
# print(validation_data[0])
# print(len(test_data[0]))
# print(test_data[0])

# print(len(training_data[0][0]))
# print(training_data[0][0])
# print(len(validation_data[0][0]))
# print(validation_data[0][0])
# print(len(test_data[0][0]))
# print(test_data[0][0])

# print(training_data[1])
# print(validation_data[1])
# print(test_data[1])

def show_data(in_list, datapoint_index, image_index):
    '''
     :param in_list: The list of datapoint (image,*)
     :param datapoint_index: The index of the datapoint to use
     :param image_index: The index of the image of the datapoint
     :return: None
    '''

    dig_img = in_list[datapoint_index][image_index].reshape(28, 28)
    plt.imshow(dig_img, interpolation='nearest')
    plt.show()


for i in range(10):
    print(i)
    show_data(training_data, 0, i)
    show_data(validation_data, 0, i)
    show_data(test_data, 0, i)




