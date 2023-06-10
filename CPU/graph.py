import sys
import matplotlib.pyplot as plt
import time

def plot_times(name, x, y):
    plt.plot(x, y, marker='o')
    plt.xlabel("Number of threads")
    plt.ylabel("Speedup")
    plt.xticks(x)
    plt.savefig(name+".png", dpi = 300)
    return

def save_results(list, filename):
    file = open(filename,'a')
    for item in list:
        file.write(str(item) + "\n")
    file.write("-----\n")
    file.close()

if __name__ == '__main__':

    NRUNS = 1
    avg_run_times = []

    len = 0
    with open("output_"+sys.argv[1]+".txt",'r') as file:
        for line in file:
            len += 1
            avg_run_times.append(float(line))
    file.close()
    n_threads_list = [elem+1 for elem in range(len)]
    speedup = [avg_run_times[0]/elem for elem in avg_run_times]
    plot_times(sys.argv[1], n_threads_list, speedup)