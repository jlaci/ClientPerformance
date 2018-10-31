import numpy as np
import matplotlib.pyplot as plt
import math as math
import random as random
from sklearn import linear_model


# Import and visualize the data
raw_data = np.genfromtxt('data/client_performance.csv', delimiter=";", dtype=float, encoding="utf-8-sig")
number_of_measurements = len(raw_data[0])


# Settings
min_history = 2
max_history = 5

# Format the data into feature and values
data = []

# Data array will contain data from each history length
for history_length in range(min_history + 1, max_history + 2):
    history_data = []
    for measurement in range(number_of_measurements):
        measurement_data = raw_data[:, measurement]
        for value_to_predict in range(history_length, len(measurement_data)):
            history_data.append(measurement_data[value_to_predict - history_length:value_to_predict])
    data.append(history_data)


# Learning
training_percentage = 0.5

for data_set in data:
    random.shuffle(data_set)
    cutting_point = math.floor(len(data_set) * training_percentage)
    training_set = np.asarray(data_set[:cutting_point])
    validation_set = np.asarray(data_set[cutting_point:])

    value_column = len(training_set[0]) - 1
    model = linear_model.LinearRegression()
    model.fit(training_set[:, list(range(0, value_column))], training_set[:, value_column])
    prediction = model.predict(validation_set[:, list(range(0, value_column))])

    # Print the accuracy
    sum_error = 0
    sum_data = 0
    for i in range(len(validation_set)):
        sum_error += math.fabs(prediction[i] - validation_set[i][value_column])
        sum_data += validation_set[i][0]

    average_error = sum_error / len(validation_set)
    average_data = sum_data / len(validation_set)

    print("History is ", len(data_set[0]) - 1)
    print("Average error is %f (%f%%)" % (average_error, average_error / average_data * 100))

    # Plot the result
    fig = plt.figure()
    ax = plt.axes()
    x_validate = range(len(prediction))

    # Plot predicted
    ax.plot(x_validate, prediction)

    # Plot actual
    ax.plot(x_validate, validation_set[:, value_column])


