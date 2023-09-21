import os
import json
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 2

models = [
        "gcn",
        "gcnii",
        "graphsage"
        ]
graphs = [
        "reddit",
        "flickr",
        #"physics"
        ]
gpus = [
        1, 2, 4, 8, 16
        ]
markers = [
        "o",
        "^",
        "+",
        "*"
        ]

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

if __name__ == "__main__":
    ms = 10

    fig, ax = plt.subplots(figsize=(6, 2.5))
    idx = 0
    for graph in graphs:
        runtimes = []
        for i in gpus:
            result_file = "./%s/%s.txt" % (
                    graph, i
                    )
            runtime = get_epoch_time(result_file)
            runtimes.append(runtime)
        plt.plot(gpus, runtimes, "-" + markers[idx], label = graph, markersize = ms)
        idx += 1
    plt.legend()
    plt.xscale("log")
    plt.xticks([], [])
    plt.savefig("runtime.pdf")
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 2.5))
    idx = 0
    for graph in graphs:
        communciations = []
        for i in gpus:
            result_file = "./%s/%s.txt" % (
                    graph, i
                    )
            comm = get_communication_volume(result_file, "Total")
            comm /= float(i)
            communciations.append(comm)
        plt.plot(gpus, communciations, "-" + markers[idx], label = graph, markersize = ms)
        idx += 1
    plt.legend()
    plt.xscale("log")
    plt.xticks([], [])
    plt.savefig("communciation.pdf")
    plt.show()






