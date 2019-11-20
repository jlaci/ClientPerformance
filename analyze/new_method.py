import numpy as np
import math as math
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn import linear_model
from sklearn import neural_network
from math import exp


# Parameters
# Chain
tr_per_block = 10
hash_per_block = 4096

# Global
known_history = 10  # The number of previous observed values
# Machine Learning
feature_vector_length = 3  # How many previous values to consider for prediction (i.e. the length of the feature vector)
refit_after_every = 5  # How often do we refit our model
# Statistical
max_history = 3  # The maximum amount of previous values to consider for prediction (if -1, all will be considered)
# RobbinsMonroe
rb_history = 5

# Import the data
raw_data = np.genfromtxt('data/fore_and_background_merge_desktop.csv', delimiter=";", dtype=float, encoding="utf-8-sig")
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


def predict_old_way(data_vector):
    # Probability of node contributing
    count_contribute = 0
    count_sleep = 0

    sum = 0
    for data_point in data_vector:
        sum += data_point

        # Calcualte the probability of contribution
        if data_point == 0:
            count_sleep += 1
        else:
            count_contribute += 1

    avg = sum / len(data_vector)
    p_contribute = count_contribute  / (count_contribute + count_sleep)
    x_i = avg * p_contribute
    return x_i


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
        prediction += data_vector[i + n + 1] * w[i]  # TODO: miért kell a + 1?!
    return prediction


def predict_rbm(k, w, data):
    prediction = 0
    for i in range(rb_history):
        prediction += w[i] * data[k - rb_history:k][i]
    return prediction


def format_for_ml(data_vector):
    # Format the data into feature and values for machine learning based predictors (with a sliding window over the data)
    ml_data = []
    # Start from the beginning of the data (we can first predict the end of the first data window)
    for value_to_predict in range(feature_vector_length, len(data_vector) - 1):
        ml_data.append(data_vector[value_to_predict - feature_vector_length:value_to_predict + 1])

    return np.asarray(ml_data)


def train_ml(data_vector):
    ml_data = format_for_ml(data_vector)
    value_column = len(ml_data[0]) - 1
    training_data = ml_data[:, list(range(0, value_column))]

    model = linear_model.LinearRegression()
    model.fit(training_data, ml_data[:, value_column])
    return model


def predict_ml(data_vector, model):
    validation_data = data_vector[len(data_vector) - feature_vector_length:].reshape(1, -1)
    return model.predict(validation_data)


def train_mlps(data_vector):
    ml_data = format_for_ml(data_vector)
    value_column = len(ml_data[0]) - 1
    training_data = ml_data[:, list(range(0, value_column))]

    #model = neural_network.MLPRegressor(learning_rate='invscaling', learning_rate_init=0.0005, power_t=0.7)
    #model = neural_network.MLPRegressor(solver='sgd', learning_rate='invscaling', activation='logistic', learning_rate_init=0.03, power_t=0.1)
    model = neural_network.MLPRegressor(hidden_layer_sizes=(100, 100, ), solver='sgd', learning_rate='invscaling', activation='logistic', learning_rate_init=0.03, power_t=0.1)

    model.fit(training_data, ml_data[:, value_column])
    return model


def update_mlpr(data_vector, model):
    ml_data = format_for_ml(data_vector)
    value_column = len(ml_data[0]) - 1
    training_data = ml_data[:, list(range(0, value_column))]
    return model.partial_fit(training_data, ml_data[:, value_column])


sum_data = 0
mse_old = 0
mse_st = 0
mse_rbm = 0
mse_ml = 0
mse_ffn = 0
se_old = 0
se_st = 0
se_rbm = 0
se_ml = 0
se_ffn = 0
n = 0

preds_baseline = []
preds_st = []
preds_rb = []
preds_ml = []
preds_ffn = []

# Simulate the time
for data_set in data_sets:
    # Starter model for ML
    model_ml = None

    # Make a calculation predicting the whole timeline, based on the previous data
    baseline = predict_old_way(data_set[:known_history])

    # Initialize RobbinsMonroe weights
    w = [1/rb_history for x in range(rb_history)]

    # Initialize the FFN
    model_ffn = train_mlps(data_set[:known_history])

    for k in range(known_history, len(data_set) - 1):
        n += 1
        sum_data += data_set[k]
        if model_ml is None or k % refit_after_every == 0:
            model_ml = train_ml(data_set)
            #print('Retrained ML')

        # Old way
        err_old = baseline - data_set[k + 1]
        se_old += math.fabs(err_old)
        mse_old += pow(err_old, 2)

        # Statistical
        pred_st = predict_statistical(data_set[:k])
        err_st = pred_st - data_set[k + 1]
        se_st += math.fabs(err_st)
        mse_st += pow(err_st, 2)
        #print(k, 'ST Prediction:', pred_st, 'actual:', data_set[k + 1], 'error:', math.fabs(pred_st - data_set[k + 1]))

        # Robbins Monroe
        # Copy the previous state of the weight
        w_k = w.copy()

        # Update each w component based on the new data
        for u in range(rb_history):
            sumv = 0
            for v in range(rb_history):
                sumv += w_k[v] * data_set[k - v]
            delta = 0.000000000000005
            w[u] = w_k[u] - delta * (data_set[k] - sumv) * data_set[k - u]
        pred_rbm = predict_rbm(k, w, data_set)
        err_rbm = pred_rbm - data_set[k + 1]
        se_rbm += math.fabs(err_rbm)
        mse_rbm += pow(err_rbm, 2)

        # Machine learning
        pred_ml = predict_ml(data_set[:k], model_ml)
        err_ml = pred_ml - data_set[k + 1]
        se_ml += math.fabs(err_ml)
        mse_ml += pow(err_ml, 2)

        # FFN
        row = data_set[k - known_history:k]
        # Predict
        pred_ffn = predict_ml(data_set[:k], model_ffn)
        err_ffn = pred_ffn - data_set[k + 1]
        se_ffn += math.fabs(err_ffn)
        mse_ffn += pow(err_ffn, 2)
        # Update weights
        model_ffn = update_mlpr(data_set[k - known_history:k], model_ffn)

        # Print predicrtions
#        print(baseline, '\t', pred_st, '\t', pred_ml[0], '\t', pred_rbm, '\t', data_set[k + 1])
        preds_baseline.append(math.fabs(data_set[k + 1] - baseline))
        preds_st.append(math.fabs(data_set[k + 1] - pred_st))
        preds_rb.append(math.fabs(data_set[k + 1] - pred_rbm))
        preds_ml.append(math.fabs(data_set[k + 1] - pred_ml[0]))
        preds_ffn.append(math.fabs(data_set[k + 1] - pred_ffn))

        print(math.fabs(data_set[k + 1] - baseline), '\t', math.fabs(data_set[k + 1] - pred_st), '\t', math.fabs(data_set[k + 1] - pred_ml[0]), '\t', math.fabs(data_set[k + 1] - pred_rbm), '\t', (math.fabs(data_set[k + 1] - pred_ffn)))

print('RMSE baseline:', math.sqrt(mse_old/n), '\tStatistical', math.sqrt(mse_st/n), '\tRobbinsMonroe', math.sqrt(mse_rbm/n), '\tLinearRegression', math.sqrt(mse_ml/n), '\tFFN', math.sqrt(mse_ffn/n))
print('AVG error, baseline:', (se_old/n)/(sum_data/n), '\tStatistical', (se_st/n)/(sum_data/n), '\tRobbinsMonroe', (se_rbm/n)/(sum_data/n), '\tLinearRegression', (se_ml/n)/(sum_data/n), '\tFFN', (se_ffn/n)/(sum_data/n))

data = [go.Histogram(x=preds_baseline)]
#py.iplot(data, filename='basic histogram')