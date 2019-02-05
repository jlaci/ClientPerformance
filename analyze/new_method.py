import numpy as np
import matplotlib.pyplot as plt
import math as math
import random as random
from sklearn import linear_model, neural_network

# Parameters
# Global
known_history = 10  # The number of previous observed values
# Machine Learning
feature_vector_length = 3  # How many previous values to consider for prediction (i.e. the length of the feature vector)
# Statistical
max_history = -1  # The maximum amount of previous values to consider for prediction (if -1, all will be considered)

# Import the data
raw_data = np.genfromtxt('data/client_performance.csv', delimiter=";", dtype=float, encoding="utf-8-sig")
number_of_measurements = len(raw_data[0])


# Format the data into feature and values for machine learning based predictors (with a sliding window over the data)
ml_data = []

data_window_length = feature_vector_length * 2
history_data = []
for measurement in range(number_of_measurements):
    measurement_data = raw_data[:, measurement]

    # Start from the beginning of the data (we can first predict the end of the first data window)
    for value_to_predict in range(data_window_length, len(measurement_data) - 1):
        history_data.append(measurement_data[value_to_predict - data_window_length:value_to_predict + 1])
ml_data.append(history_data)



