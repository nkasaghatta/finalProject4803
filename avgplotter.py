import numpy as np
import matplotlib.pyplot as plt


def plot_data(filename):
    with open(filename, 'r') as file:
        y_values = np.array([float(line.strip()) for line in file])

    chunk_size = 5000
    num_chunks = len(y_values) // chunk_size

    averages = [np.mean(y_values[i * chunk_size:(i + 1) * chunk_size]) for i in range(num_chunks)]

    x_values = np.arange(chunk_size / 2, len(y_values), chunk_size)

    # Plotting the data
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(x_values, averages, marker='o', linestyle='-', linewidth=1)
    plt.title('Average Value Every 5000 Data Points')
    plt.xlabel('Index (x)')
    plt.ylabel('Average Value (y)')
    plt.grid(True)
    plt.show()


plot_data('ICPC_2_ML_6_SL_500_SC_10/icpcdata.txt')
