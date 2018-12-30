import numpy as np
import matplotlib.pyplot as plt
import math as math
import random as random
from sklearn import linear_model, neural_network

# Import and visualize the data
raw_data = np.genfromtxt('data/client_performance.csv', delimiter=";", dtype=float, encoding="utf-8-sig")
number_of_measurements = len(raw_data[0])

# Visualize on plot
fig = plt.figure()
ax = plt.axes()
x = range(len(raw_data))

for client in range(number_of_measurements):
    ax.plot(x, [i[client] for i in raw_data])

# Settings, history is the values considered before the current, eg.: in case of 1, 3, # the history length is 2
min_history = 2
max_history = 20

# Format the data into feature and values
data = []

# Data array will contain data from each history length (these will be inspected separately)
for history_length in range(min_history, max_history):
    # Data window is the current data plus history,
    data_window_length = history_length * 2
    history_data = []
    for measurement in range(number_of_measurements):
        measurement_data = raw_data[:, measurement]

        # Start from the beginning of the data (we can first predict the end of the first data window)
        for value_to_predict in range(data_window_length, len(measurement_data) - 1):
            history_data.append(measurement_data[value_to_predict - data_window_length:value_to_predict + 1])
    data.append(history_data)


def autocorrellation(data_vector, history_length, lag):

    sum = 0
    for k in range(len(data_vector) - history_length, len(data_vector)):
        sum += data_vector[k] * data_vector[k - lag]
    return sum / history_length


def autocorrellation2(data_vector, history_length, lag):
    if lag == 0:
        return 1
    else:
        start = len(data_vector) - history_length
        normal = data_vector[start:]
        shifted = data_vector[start - lag: len(data_vector) - lag]
        return np.corrcoef(normal, shifted)[0, 1]


def calculate_weights(data_matrix, n):
    average_weights = [0] * n

    for data_vector in data_matrix:
        r_matrix = [[0] * n for i in range(n)]
        r_vector = [0] * n
        for i in range(n):
            for j in range(n):
                r_matrix[i][j] = autocorrellation(data_vector, n, np.abs(j - i))
            r_vector[i] = autocorrellation(data_vector, n, i + 1)

        weights = np.linalg.solve(r_matrix, r_vector)

        for i in range(n):
            average_weights[i] += weights[i]

    for i in range(n):
        average_weights[i] = average_weights[i] / len(data_matrix)

    return average_weights


def row_weight(data_vector, n):
    r_matrix = [[0] * n for i in range(n)]
    r_vector = [0] * n
    for i in range(n):
        for j in range(n):
            r_matrix[i][j] = autocorrellation(data_vector, n, np.abs(j - i))
        r_vector[i] = autocorrellation(data_vector, n, i + 1)

    return np.linalg.solve(r_matrix, r_vector)


def predict(data_to_predict, history_length, w):
    predictions = []

    for data_row in data_to_predict:
        prediction = 0
        wr = row_weight(data_row, history_length)
        for i in range(history_length):
            prediction += data_row[i + (len(data_row) - history_length)] * wr[i]
        predictions.append(prediction)

    return predictions


# Learning with Linear Regression
training_percentage = 0.5
runs = 1
m1_errors = [0 for k in range(len(data))]
m2_errors = [0 for k in range(len(data))]
m3_errors = [0 for k in range(len(data))]

for j in range(runs):
    for i, data_set in enumerate(data):
        # Shuffle the data points (vectors, not individual measurements!)
        random.shuffle(data_set)

        cutting_point = math.floor(len(data_set) * training_percentage)
        training_set = np.asarray(data_set[:cutting_point])
        value_column = len(training_set[0]) - 1
        training_data = training_set[:, list(range(0, value_column))]
        validation_set = np.asarray(data_set[cutting_point:])
        validation_data = validation_set[:, list(range(0, value_column))]

        history_length = min_history + i

        # Linear predictor with statistical weights
        w = calculate_weights(training_data, history_length)
        m1_prediction = predict(validation_data, history_length, w)

        # Linear Regression model
        model = linear_model.LinearRegression()
        model.fit(training_data, training_set[:, value_column])
        m2_prediction = model.predict(validation_data)

        # Print the accuracy
        m1_sum_error = 0
        m2_sum_error = 0
        n = len(validation_set)
        for k in range(n):
            m1_sum_error += math.pow(m1_prediction[k] - validation_set[k][value_column], 2)
            m2_sum_error += math.pow(m2_prediction[k] - validation_set[k][value_column], 2)

        m1_error = math.sqrt(m1_sum_error / n)
        m2_error = math.sqrt(m2_sum_error / n)

        m1_errors[i] += m1_error
        m2_errors[i] += m2_error


        #print("History is ", history_length)
        #print("M1 Average error is %f" % m1_error)
        #print("M2 Average error is %f" % m2_error)

        # Plot the result
        #fig = plt.figure()
        #ax = plt.axes()
        #x_validate = range(len(m2_prediction))

        # Plot predicted
        #ax.plot(x_validate, m2_prediction)

        # Plot actual
        #ax.plot(x_validate, validation_set[:, value_column])

print("M1 Average error is")
for i in range(len(data)):
    m1_avg = m1_errors[i]/runs
    print("%f" % m1_avg)

print("M2 Average error is")
for i in range(len(data)):
    m2_avg = m2_errors[i]/runs
    print("%f" % m2_avg)