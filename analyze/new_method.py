import numpy as np
import matplotlib.pyplot as plt
import math as math
import random as random
from sklearn import linear_model

# Parameters
# Global
known_history = 10  # The number of previous observed values
# Machine Learning
feature_vector_length = 3  # How many previous values to consider for prediction (i.e. the length of the feature vector)
refit_after_every = 5  # How often do we refit our model
# Statistical
max_history = 3  # The maximum amount of previous values to consider for prediction (if -1, all will be considered)

# Import the data
raw_data = np.genfromtxt('data/client_performance.csv', delimiter=";", dtype=float, encoding="utf-8-sig")
number_of_measurements = len(raw_data[0])

# Parse the data file to data sets
data_sets = []
for measurement in range(number_of_measurements):
    data_sets.append(raw_data[:, measurement])
print('Parsing ok')


def autocorrellation(data_vector, n, lag):
    sum = 0
    for k in range(len(data_vector) - n, len(data_vector)):
        sum += data_vector[k] * data_vector[k - lag]
    return sum / n


def predict_statistical(data_vector):
    # Whats the length of previous data to consider
    n = math.floor(len(data_vector) / 2)

    # If we limit the maximum history
    if max_history != -1 and n > max_history:
        n = max_history

    # Calculate weights
    r_matrix = [[0] * n for i in range(n)]
    r_vector = [0] * n
    for i in range(n):
        for j in range(n):
            r_matrix[i][j] = autocorrellation(data_vector, n, np.abs(j - i))
        r_vector[i] = autocorrellation(data_vector, n, i + 1)
    w = np.linalg.solve(r_matrix, r_vector)

    # Predict with the weights
    prediction = 0
    for i in range(n):
        prediction += data_vector[i + n] * w[i]
    return prediction


def train_ml(data_vector):
    # Format the data into feature and values for machine learning based predictors (with a sliding window over the data)
    ml_data = []
    # Start from the beginning of the data (we can first predict the end of the first data window)
    for value_to_predict in range(feature_vector_length, len(data_vector) - 1):
        ml_data.append(data_vector[value_to_predict - feature_vector_length:value_to_predict + 1])

    ml_data = np.asarray(ml_data)
    value_column = len(ml_data[0]) - 1
    training_data = ml_data[:, list(range(0, value_column))]

    model = linear_model.LinearRegression()
    model.fit(training_data, ml_data[:, value_column])
    return model


def predict_ml(data_vector, model):
    validation_data = data_vector[len(data_vector) - feature_vector_length:].reshape(1, -1)
    return model.predict(validation_data)


# Simulate the time
for data_set in data_sets:
    model_ml = None
    for k in range(known_history, len(data_set) - 1):
        if model_ml is None or k % refit_after_every == 0:
            model_ml = train_ml(data_set)
            #print('Retrained ML')

        pred_st = predict_statistical(data_set[:k])
        #print(k, 'ST Prediction:', pred_st, 'actual:', data_set[k + 1], 'error:', math.fabs(pred_st - data_set[k + 1]))

        pred_ml = predict_ml(data_set[:k], model_ml)
        #print(k, 'ML Prediction:', pred_ml, 'actual:', data_set[k + 1], 'error:', math.fabs(pred_ml - data_set[k + 1]))
        print(pred_st, '\t', pred_ml[0], '\t', data_set[k + 1])

