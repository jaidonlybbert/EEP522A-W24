import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


def numpy_gemm(max_size):
    iterations = int(max_size / 16)
    sizes = [x*16 for x in range(1, iterations)]
    times = []
    for n in sizes:
        times.append(test_gemm(n))
    data = {'Matrix Size': sizes, 'Execution Time': times}
    print(data)
    return data


def plot_numpy_gemm():
    data = numpy_gemm(2000)
    plt.figure(figsize=(10, 6))
    plt.plot(data['Matrix Size'], data['Execution Time'], color='red', label="Numpy")
    plt.title('CPU Numpy GEMM Performance')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (ms)')
    plt.savefig('gemm_numpy.png')

def test_gemm(n):
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    start = time.perf_counter()
    c = a @ b
    end = time.perf_counter()
    return (end - start) * 1E3


def plot_csvs(file1, file2):
    # Read the CSV files
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    data3 = numpy_gemm(200)

    # Create the scatter plots
    plt.figure(figsize=(10, 6))
    plt.plot(data1['Matrix Size'], data1['Execution Time'], color='blue', label=file1)
    plt.plot(data2['Matrix Size'], data2['Execution Time'], color='red', label=file2)
    plt.plot(data3['Matrix Size'], data3['Execution Time'], color='red', label="Numpy")
    plt.title('CPU GEMM Execution Time Comparison')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (ms)')
    plt.legend()
    plt.savefig('gemm.png')

def plot_compute_bound():
    # Read the CSV file
    data = pd.read_csv('gemm_benchmark.csv', index_col=False)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['Matrix Size'], data['Execution Time'], marker='o')
    plt.title('GEMM Execution Time (compute bound)')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (ms)')


def plot_matrix_add():
    # Read the CSV file
    data = pd.read_csv('matrix_add_benchmark.csv', index_col=False)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Matrix Size'], data['Execution Time'], marker='o')
    plt.title('Matrix Add Execution Time (compute bound)')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (ms)')


def plot_tiled_gemm_cpu():
    # Read the CSV file
    data = pd.read_csv('tiled_gemm_benchmark.csv', index_col=False)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Matrix Size'], data['Execution Time'], marker='o')
    plt.title('Tiled GEMM Execution Time')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (ms)')


def plot_memory_bound():
    # Read the CSV file
    data = pd.read_csv('gemm_benchmark_memory.csv', index_col=False)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Matrix Size'], data['Execution Time'], marker='o')
    plt.title('GEMM Execution Time')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (us)')


if __name__ == "__main__":
    #plot_compute_bound()
    #plot_memory_bound()
    #plot_matrix_add()
    #plot_tiled_gemm_cpu()
    plot_csvs("gemm_benchmark.csv", "tiled_gemm_benchmark.csv")
    plot_numpy_gemm()
    plt.show()
