import numpy as np
import math

raw_data = np.genfromtxt('data/time-window/tw1.csv', delimiter=";", dtype=float, encoding="utf-8-sig")
original_delta = 100    # Original time window used in the measurement file

number_of_measurements = len(raw_data[0])

##########################################################################################
# Global parameters
known_history = 10      # The number of previous observed values used for prefiction
delta_min = 100         # Minimum time window size
delta_step = 100        # Increment of time window size
delta_max = 1000        # Maximum of the time window size

# Optimalization parameters
alpha_min = 0
alpha_max = 100
alpha_step = 1

# Measurement overhead (% of the total client performance contribution)
network_time = 100  # The time it takes to upload the result to the server
processing_time = 10  # Number of milliseconds it takes for the server to receive process the anwer
prediction_time = 0.1  # milliseconds for prediction

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


# Returns the measurement overhead for the given delta. The overhead is the time when a client measures and sends the
# data instead of calculating. Here we calculate the lost performance per second
def measurement_overhead(delta, data_set):
    # Calculate the average performance / second
    performance_per_second = np.average(data_set) * (1000 / original_delta)
    measurement_per_second = 1000 / delta

    # The cost of a single mesasurement in seconds
    measurement_cost = (processing_time + prediction_time) / 1000

    # The total number of measurements is total length (in seconds) * measurement / second * measurement_cost
    total_measurement_time = ((len(data_set) * original_delta) / 1000) * measurement_per_second * measurement_cost

    # Lost performance is the amount of measurements
    lost_performance = total_measurement_time * performance_per_second
    max = np.amax(data_set) * (1000 / original_delta)
    # Normalized lost performance
    return lost_performance / max

# Mérések lefuttatása
best_overall_delta = delta_min
best_overall_delta_value = -1
best_overall_alpha_value = -1.0

for delta in range(delta_min, delta_max, delta_step):
    remaining_error = 0
    total_overhead = 0
    for data_set in data_sets:
        data_set = convert_data_set(data_set, original_delta, delta)
        overhead = measurement_overhead(delta, data_set)
        normalize(data_set)

        # The data_vector is the known part of the dataset
        data_vector = data_set[:known_history]
        r_matrix = calculate_r_matrix(data_vector)
        r_vector = calculate_r_vector(data_vector)
        r_inverse_matrix = np.linalg.inv(r_matrix)
        r_vector_transposed = np.array(r_vector).T

        r_0 = data_set[known_history + 1] * data_set[known_history + 1]  # E(x^2)
        rRr = np.matmul(np.matmul(r_vector_transposed, r_matrix), r_vector)  # r
        remaining_error += math.fabs(r_0 - rRr)
        total_overhead += overhead

    # Calculate the best delta for each measurement
    avg_remaining_error = remaining_error / len(data_sets)
    avg_overhead = total_overhead / len(data_sets)

    for alpha in range(alpha_min, alpha_max, alpha_step):
        weighted_cost = (1 - alpha/alpha_max) * avg_remaining_error + (alpha/alpha_max) * avg_overhead
        if best_overall_delta_value == -1 or weighted_cost < best_overall_delta_value:
            best_overall_delta_value = weighted_cost
            best_overall_delta = delta
            best_overall_alpha_value = alpha / alpha_max
        #print('Delta', delta, 'avg error:', avg_remaining_error, 'overhead: ',  avg_overhead, 'weighted error: ', weighted_cost, 'alpha: ', alpha/alpha_max)
        print(delta/1000, alpha/alpha_max, weighted_cost)


print('Weighted best delta', best_overall_delta, ':', best_overall_delta_value)
print('Weighted best delta', best_overall_delta, ':', best_overall_delta_value, ' with alpha: ', best_overall_alpha_value)
