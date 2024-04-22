import numpy as np
import matplotlib.pyplot as plt


def plot_data(filename, outputPlot):
    with open(filename, 'r') as file:
        y_values = [float(line.strip()) for line in file]

    x_values = np.arange(1, len(y_values) + 1)

    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(x_values, y_values, marker='.', linestyle='-', markersize=1, linewidth=0.5)
    plt.title('Plot of Data Points')
    plt.xlabel('Index (x)')
    plt.ylabel('Value (y)')
    plt.grid(True)

    plt.savefig(outputPlot)
    plt.close()

# You can call this function with the path to your file
plot_data('ICPC_2_ML_6_SL_500_SC_10/icpcdata.txt', 'ICPC_2_ML_6_SL_500_SC_10/fullPlot.png')
