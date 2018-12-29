import numpy as np

# Import and visualize the data
raw_data = np.genfromtxt('data/client_performance.csv', delimiter=";", dtype=float, encoding="utf-8-sig")
number_of_measurements = len(raw_data[0])

# History length is the number of previous time steps we consider, therefore the number of components the w vector has
history_length = 5
data = []
delta = 1


def predict(k, w, data):
    prediction = 0
    for i in range(history_length):
        prediction += w[i] * data[k - history_length:k][i]
    return prediction


# Each measurement is a clients session
for measurement in range(number_of_measurements):
    w = [1/history_length for x in range(history_length)]
    data = raw_data[:, measurement]

    # For each time step where k is the last known data
    for k in range(history_length, len(data) - 1):
        print('---------- k = %d ----------' % k)
        print('w:', w)
        print('prediction:', predict(k, w, data))
        print('actual:', data[k + 1])
        print('error:', abs(predict(k, w, data) - data[k + 1]))

        # Copy the previous state of the weight
        w_k = w.copy()

        # Update each w component based on the new data
        for u in range(history_length):
            sumv = 0
            for v in range(history_length):
                sumv += w_k[v] * data[k - v]
            delta = 0.0000001
            w[u] = w_k[u] - delta * (data[k] - sumv)
