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

markers = [
        "o",
        "^",
        "+",
        "*"
        ]
colors = [
        "orange",
        "green",
        "red",
        "steelblue"
        ]
markers = [
        "o",
        "^",
        "+",
        "*"
        ]
ms = 10

models = [
        "gcn",
        "graphsage",
        "gcnii",
        "resgcn"
        ]
model_names = [
        "GCN",
        "GraphSage",
        "GCNII",
        "ResGCN+"
        ]
gpus = [
        2, 4, 8, 16
        ]

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
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_xscale('log', base=2)

    for model_idx in range(len(models)):
        model = models[model_idx]
        # graph parallel
        runtimes = []
        used_gpus = gpus
        if model == "gcnii" or model == "resgcn":
            used_gpus = [4, 8, 16]
        for gpu in used_gpus:
            result_file = "./graph/%s/%s.txt" % (
                    model, gpu
                    )
            epoch_time = get_epoch_time(result_file)
            runtimes.append(epoch_time)
        plt.plot(
                used_gpus, runtimes, "-" + markers[model_idx],
                color = colors[model_idx], 
                label = model_names[model_idx] + " (Graph Parallel)",
                markersize = ms
                )

        runtimes = []
        used_gpus = gpus
        for gpu in used_gpus:
            result_file = "./pipeline/%s/%s.txt" % (
                    model, gpu
                    )
            epoch_time = get_epoch_time(result_file)
            runtimes.append(epoch_time)
        plt.plot(
                used_gpus, runtimes, "--" + markers[model_idx],
                color = colors[model_idx], 
                label = model_names[model_idx] + " (Pipeline Parallel)",
                markersize = ms
                )

    plt.xticks(gpus, gpus)
    
    plt.xlabel("NumGPU")
    plt.ylabel("Per-Epoch Time (s)")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("strong_scaling.pdf")
    plt.show()





