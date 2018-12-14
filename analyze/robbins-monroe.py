import numpy as np

# Import and visualize the data
raw_data = np.genfromtxt('data/client_performance.csv', delimiter=";", dtype=float, encoding="utf-8-sig")
number_of_measurements = len(raw_data[0])

# Format the data into feature and values
history_length = 5
data = []

for measurement in range(number_of_measurements):
    w = [0 for x in range(history_length)]
    data = raw_data[:, measurement]
    for k in range(len(data)):
        new_w = [0 for x in range(history_length)]
        # Update each w component
        for u in range(history_length):
            sumv = 0
            for v in range(history_length):
                sumv += w[v] * data[k - history_length]
            delta = 1
            new_w[u] = w[u] - delta * (data[k] - sumv) * data[k - u]
        w = new_w
        print('w:', w)


