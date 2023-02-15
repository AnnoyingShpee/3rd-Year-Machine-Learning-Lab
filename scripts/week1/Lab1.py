import numpy
import scipy
import matplotlib
import pandas
import sklearn

# print ('numpy', numpy.__version__)
# print ('scipy', scipy.__version__)
# print ('matplotlib', matplotlib.__version__)
# print ('pandas', pandas.__version__)
# print ('sklearn', sklearn.__version__)

# Loads data from csv into variable
data = pandas.read_csv('../../data/wdbc.data', header=None)
# Prints (rows, columns)
# print(data.shape)

# Prints top 10 rows and [rows x columns]
# print(data.head(10))

# Prints the number of occurrences for each value in a specific column
# print(data[1].value_counts())

# groupby(1)[2].mean() means grouping the values of column 1 and
# getting the mean of column 2 values in each column 1 group
mean_values = data.groupby(1)[2].mean()
# mean_values[0] is the mean of B and mean_values[1] is the mean of M
threshold = (mean_values[0] + mean_values[1]) / 2
# print(mean_values)
# print('Threshold', threshold)


def predict(new_instance):
    # Reads value of column 2 in row instance
    radius = new_instance[2]
    threshold_point = 14.804676999101527

    # If radius <= threshold, Then Prediction = Benign
    # Else, Prediction = Malignant
    if radius <= threshold_point:
        return 'B'
    else:
        return 'M'


# Apply rule to the dataset
true_predictions = 0            # Number of correct predictions
number_instances = len(data)    # Total number of instances in the dataset

for i in range(len(data)):
    instance = data.loc[i]      # Read instance at row i
    instance_class = instance[1]  # Actual class. Get the actual value from column 1

    # Apply rule, create predict function
    prediction = predict(instance)
    # print('Actual', instance_class, 'Prediction', prediction)

    if instance_class == prediction:
        true_predictions += 1  # Count correct predictions

# Compute accuracy as performance metric of the model
accuracy = true_predictions / number_instances
print('No class')
print('True Predictions', true_predictions, 'out of', number_instances)
print('accuracy', accuracy)

# Extras
# Count the number of occurrences for each unique value in column 1 and getting the smallest value of the count
training_size_per_class = data[1].value_counts()[1].min()
# Get a fraction of the result and round it
training_size_per_class = round(training_size_per_class * 0.7)
# print(training_size_per_class)
# Groups the data by column 1 values and get n rows from each group
training_dataset = data.groupby(1).head(training_size_per_class)
# Remove training_dataset from original data
testing_dataset = data.drop(training_dataset.index, axis=0)
training_dataset.reset_index(inplace=True)
testing_dataset.reset_index(inplace=True)
# training_dataset.pop('index')
# testing_dataset.pop('index')


class ThresholdModel:
    def __init__(self, dataset, column_to_group, column_to_calculate):
        self.data = dataset
        self.group = column_to_group
        self.data_column = column_to_calculate

    def count_groups(self):
        return self.data[self.group].nunique()

    def get_threshold(self):
        # Group the data by column (self.group) and calculate the mean of column (self.data_column) in for each group
        mean_value = self.data.groupby(self.group)[self.data_column].mean()
        number_of_groups = self.count_groups()
        total = 0
        for x in range(number_of_groups):
            total += mean_value[x]
        return total / number_of_groups

    def get_prediction(self, row):
        value = row[2]
        threshold_value = self.get_threshold()
        if value <= threshold_value:
            return 'B'
        else:
            return 'M'

    def get_accuracy(self):
        total_correct = 0               # Total number of correct predictions
        number_of_instances = len(self.data)    # Total number of rows
        for z in range(number_of_instances):
            row = self.data.loc[z]      # Get the row at index i
            actual_value = row[1]       # Get value at column 1 in row

            predicted_value = self.get_prediction(row)
            # print('Actual', actual_value, 'Prediction', predicted_value)

            if actual_value == predicted_value:
                total_correct += 1

        model_accuracy = total_correct / number_of_instances
        print(total_correct, 'Correct Predictions out of', number_of_instances)
        print('Accuracy: ', model_accuracy)


training = ThresholdModel(training_dataset, 1, 2)
testing = ThresholdModel(testing_dataset, 1, 2)

print('Training Dataset: ')
training.get_accuracy()
print('Testing Dataset: ')
testing.get_accuracy()
