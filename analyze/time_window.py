import numpy as np
import math

raw_data = np.genfromtxt('data/fore_and_background_merge_desktop.csv', delimiter=";", dtype=float, encoding="utf-8-sig")
number_of_measurements = len(raw_data[0])

##########################################################################################
# Global parameters
known_history = 10  # The number of previous observed values
original_delta = 1000
delta_min = 1000
delta_step = 1000
delta_max = 10000
##########################################################################################

# Parse the data file to data sets
data_sets = []
for measurement in range(number_of_measurements):
    data_sets.append(raw_data[:, measurement])
print('Parsing ok')


def autocorrellation(x, lag):
    if (lag == 0):
        return 1
    else:
        return np.corrcoef(np.array([x[:-lag], x[lag:]]))[0][1]


def calculate_r_matrix(data_vector):
    n = int(known_history / 2)

    r_matrix = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            r_matrix[i][j] = autocorrellation(data_vector, np.abs(j - i))
    return r_matrix


def calculate_r_vector(data_vector):
    n = int(known_history / 2)

    r_vector = [0] * n
    for i in range(n):
        r_vector[i] = autocorrellation(data_vector, i + 1)
    return r_vector


def normalize(vector):
    max = np.amax(vector)
    for i in range(len(vector)):
        vector[i] = vector[i] / max


# Converts the given dataset to the target delta
def convert_data_set(data_set, original_delta, delta):
    if delta % original_delta != 0:
        raise Exception('Delta must be the multiple of original delta')

    ratio = int(delta / original_delta)
    result = [0] * math.ceil(len(data_set) / ratio)
    for i in range(len(data_set)):
        result[int(i/ratio)] += data_set[i]

    return result


for delta in range(delta_min, delta_max, delta_step):
    remaining_error = 0
    for data_set in data_sets:
        data_set = convert_data_set(data_set, original_delta, delta)
        normalize(data_set)

        # The data_vector is the known part of the dataset
        data_vector = data_set[:known_history]
        r_matrix = calculate_r_matrix(data_vector)
        r_vector = calculate_r_vector(data_vector)
        r_inverse_matrix = np.linalg.inv(r_matrix)
        r_vector_transposed = np.array(r_vector).T

        r_0 = data_set[known_history + 1]
        rRr = np.matmul(np.matmul(r_vector_transposed, r_matrix), r_vector)
        remaining_error += math.fabs(r_0 * r_0 - rRr)

    print('Delta', delta, ':', remaining_error / len(data_sets))
