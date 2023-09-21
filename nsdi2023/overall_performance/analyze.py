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
        "reddit",
        "physics"
        ]
models = [
        "gcn",
        "graphsage",
        "gcnii",
        "resgcn"
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
    avg_speedups = 1.
    max_seppedup = 0
    for graph in graphs:
        for model in models:
            epoch_times = []
            graph_t = None
            pipeline_t = None
            for method in methods:
                s = 0.
                for seed in range(1, num_runs + 1):
                    result_file = "./results/%s/%s/%s/%s.txt" % (
                            method, graph, model, seed
                            )
                    t = get_epoch_time(result_file)
                    s += t
                s /= float(num_runs)
                if method == "graph":
                    graph_t = s
                elif method == "pipeline":
                    pipeline_t = s
                epoch_times.append("%.2f" % (s))
            assert(graph_t != None)
            assert(pipeline_t != None)
            avg_speedups *= (graph_t / pipeline_t)
            max_seppedup = max(max_seppedup, graph_t / pipeline_t)
            print(graph, model, epoch_times, "%.2f" % (graph_t / pipeline_t))
    avg_speedups = avg_speedups ** (1. / float(len(graphs) * len(methods)))
    print("Average / MAX improvement: %.2f / %.2f" % (avg_speedups, max_seppedup))

def print_accuracies():
    print("Accuracy (%)")
    print(methods)
    for graph in graphs:
        for model in ["gcnii", "resgcn"]:
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
    avg_improv = 1.
    max_improv = 0.
    for graph in graphs:
        for model in models:
            epoch_times = []
            graph_v = None
            pipeline_v = None 
            for method in methods:
                s = 0.
                for seed in range(1, num_runs + 1):
                    result_file = "./results/%s/%s/%s/%s.txt" % (
                            method, graph, model, seed
                            )
                    s += get_communication_volume(result_file, "Total")
                s /= float(num_runs)
                if method == "graph":
                    graph_v = s
                elif method == "pipeline":
                    pipeline_v = s
                epoch_times.append("%.2f" % (s))
            assert(graph_v != None and pipeline_v != None)
            improv = graph_v / pipeline_v
            avg_improv *= improv
            max_improv = max(max_improv, improv)
            print(graph, model, epoch_times, "%.2f" % (improv))
    avg_improv = avg_improv ** (1. / float(len(graphs) * len(models)))
    print("Average / MAX improvement: %.2f / %.2f" % (avg_improv, max_improv))

def print_communication_time():
    print("Communication Time (unit: ms)")
    print(methods)
    avg_improv = 1.
    max_improv = 0.
    for graph in graphs:
        for model in models:
            epoch_times = []
            graph_v = None
            pipeline_v = None 
            for method in methods:
                s = 0.
                for seed in range(1, num_runs + 1):
                    result_file = "./results/%s/%s/%s/%s.txt" % (
                            method, graph, model, seed
                            )
                    s += get_breakdown_time(result_file, "Communication-Layer")
                    s += get_breakdown_time(result_file, "Communication-Graph")
                s /= float(num_runs)
                if method == "graph":
                    graph_v = s
                elif method == "pipeline":
                    pipeline_v = s
                epoch_times.append("%.2f" % (s))
            assert(graph_v != None and pipeline_v != None)
            improv = graph_v / pipeline_v
            avg_improv *= improv
            max_improv = max(max_improv, improv)
            print(graph, model, epoch_times, "%.2f" % (improv))
    avg_improv = avg_improv ** (1. / float(len(graphs) * len(models)))
    print("Average / MAX improvement: %.2f / %.2f" % (avg_improv, max_improv))

def plot_convergence_curves():
    model_name = ["GCNII", "ResGCN+"]
    for graph in graphs:
        model_idx = 0
        for model in ["gcnii", "resgcn"]:
            fig, ax = plt.subplots(figsize=(4.2, 2.5))
            for method in ["graph", "pipeline"]:
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
                plt.title("%s-%s" % (graph, model_name[model_idx]))
                #plt.xlabel("Epoch")
                #plt.ylabel("Test Accuracy")
            #max_y = max(y)
            #plt.ylim([max_y * 0.5, max_y * 1.1])
            if graph == "reddit":
                plt.ylim([0.85, 0.98])
            if graph == "squirrel" and model == "gcnii":
                plt.legend()
            plt.savefig("convergence_%s_%s.pdf" % (graph, model))
            plt.show()
            model_idx += 1

if __name__ == "__main__":
    print_runtimes()
    print_communication_volume()
    print_communication_time();
    print_accuracies();
    plot_convergence_curves()



