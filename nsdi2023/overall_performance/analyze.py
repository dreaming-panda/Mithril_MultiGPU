import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib import style
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2

graphs = [
        "squirrel",
        "flickr",
        "reddit"
        ]
models = [
        "gcn",
        "graphsage",
        "gcnii"
        ]
methods = [
        "graph",
        "pipeline",
        "hybrid"
        ]
num_runs = 3

def get_epoch_time(result_file):
    with open(result_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if "****** Epoch Time (Excluding Evaluation Cost):" in line:
                line = line.strip().split(" ")
                t = float(line[-3])
                return t
    assert(False)

def get_breakdown_time(result_file, breakdown_name):
    with open(result_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if ("Cluster-Wide Average, %s" % (breakdown_name)) in line:
                line = line.strip().split(" ")
                t = float(line[3])
                return t
    assert(False)

def get_communication_volume(result_file, comm_type):
    with open(result_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if ("%s communication (cluster-wide, per-epoch):" % (comm_type)) in line:
                line = line.strip().split(" ")
                t = float(line[-2])
                return t
    assert(False)

def get_test_accuracy(result_file):
    with open(result_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if "Target test_acc: " in line:
                line = line.strip().split(" ")
                t = float(line[-1])
                return t
    assert(False)

def print_runtimes():
    print("Epoch Time (unit: s)")
    print(methods)
    for graph in graphs:
        for model in models:
            epoch_times = []
            for method in methods:
                s = 0.
                for seed in range(1, num_runs + 1):
                    result_file = "./results/%s/%s/%s/%s.txt" % (
                            method, graph, model, seed
                            )
                    s += get_epoch_time(result_file)
                s /= float(num_runs)
                epoch_times.append("%.2f" % (s))
            print(graph, model, epoch_times)

def print_accuracies():
    print("Accuracy (%)")
    print(methods)
    for graph in graphs:
        for model in models:
            accuracies = []
            for method in methods:
                s = 0.
                for seed in range(1, num_runs + 1):
                    result_file = "./results/%s/%s/%s/%s.txt" % (
                            method, graph, model, seed
                            )
                    s += get_test_accuracy(result_file)
                s /= float(num_runs)
                accuracies.append("%.2f" % (s * 100.))
            print(graph, model, accuracies)

def print_communication_volume():
    print("Communication Volume (unit: GB)")
    print(methods)
    for graph in graphs:
        for model in models:
            epoch_times = []
            for method in methods:
                s = 0.
                for seed in range(1, num_runs + 1):
                    result_file = "./results/%s/%s/%s/%s.txt" % (
                            method, graph, model, seed
                            )
                    s += get_communication_volume(result_file, "Total")
                s /= float(num_runs)
                epoch_times.append("%.2f" % (s))
            print(graph, model, epoch_times)

def plot_convergence_curves():
    for graph in graphs:
        for model in ["gcnii"]:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            for method in ["graph", "pipeline", "hybrid"]:
                result_file = "./results/%s/%s/%s/1.txt" % (
                        method, graph, model
                        )
                x = []
                y = []
                with open(result_file, "r") as f:
                    while True:
                        line = f.readline()
                        if line == None or len(line) == 0:
                            break
                        if "Epoch" in line and "TestAcc" in line:
                            line = line.strip().split()
                            x.append(int(line[1][:-1]))
                            y.append(float(line[-3]))
                #print(x)
                #print(y)
                plt.plot(x, y, label = method + "-parallel")
                plt.title("%s-%s" % (graph, model))
                #plt.xlabel("Epoch")
                #plt.ylabel("Test Accuracy")
            #max_y = max(y)
            #plt.ylim([max_y * 0.5, max_y * 1.1])
            plt.legend()
            plt.show()

if __name__ == "__main__":
    print_runtimes()
    print_communication_volume()
    print_accuracies();
    plot_convergence_curves()



