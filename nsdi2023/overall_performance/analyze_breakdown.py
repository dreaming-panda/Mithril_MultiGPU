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

graphs = [
        "squirrel",
        "flickr",
        "reddit"
        ]
models = [
        "gcnii",
        "gcn",
        "graphsage"
        ]

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

def plot_breakdown(method, show_legend = True):

    print("****** Method: %s *******" % (method))

    fig, ax = plt.subplots(figsize=(10, 2))

    n_groups = 3
    index = np.arange(n_groups)
    bar_width = 0.19
    patterns = ["\\\\\\" , "///" , "...", "xxx", "O" ]

    avg_comp_ratio = 0
    avg_comm_ratio = 0
    avg_bubble_ratio = 0

    app_idx = 0
    for model in models:
        app_idx += 1

        compute_ratios = []
        bubble_ratios = []
        comm_ratios = []
        optimization_ratios = []
        others_ratios = []

        for graph in graphs:
            result_file = "./results/%s/%s/%s/1.txt" % (
                    method, graph, model
                    )
            compute_t = get_breakdown_time(result_file, "Compute")
            bubble_t = get_breakdown_time(result_file, "Bubble-Pipeline") + get_breakdown_time(result_file, "Bubble-Imbalance")
            comm_t = get_breakdown_time(result_file, "Communication-Layer") + get_breakdown_time(result_file, "Communication-Graph")
            opt_t = get_breakdown_time(result_file, "Optimization")
            others_t = get_breakdown_time(result_file, "Others")

            s = compute_t + bubble_t + comm_t + opt_t + others_t
            avg_comp_ratio += compute_t / s
            avg_comm_ratio += comm_t / s
            avg_bubble_ratio += bubble_t / s
            compute_ratios.append(compute_t / s)
            bubble_ratios.append((bubble_t + compute_t) / s)
            comm_ratios.append((comm_t + bubble_t + compute_t) / s)
            optimization_ratios.append((opt_t + comm_t + bubble_t + compute_t) / s)
            others_ratios.append((opt_t + compute_t + bubble_t + comm_t + others_t) / s)

        if app_idx == 1:
            plt.bar(
                    index + app_idx * bar_width, 
                    others_ratios,
                    bar_width,
                    label = "others",
                    edgecolor = "black",
                    color = "tomato",
                    hatch = patterns[3]
                    )
            plt.bar(
                    index + app_idx * bar_width, 
                    optimization_ratios,
                    bar_width,
                    label = "optimization",
                    edgecolor = "black",
                    color = "palegreen",
                    hatch = patterns[3]
                    )
            plt.bar(
                    index + app_idx * bar_width, 
                    comm_ratios,
                    bar_width,
                    label = "communication",
                    edgecolor = "black",
                    color = "fuchsia",
                    hatch = patterns[2]
                    )
            plt.bar(
                    index + app_idx * bar_width, 
                    bubble_ratios,
                    bar_width,
                    label = "bubble",
                    edgecolor = "black",
                    color = "aquamarine",
                    hatch = patterns[1]
                    )
            plt.bar(
                    index + app_idx * bar_width, 
                    compute_ratios,
                    bar_width,
                    label = "compute",
                    edgecolor = "black",
                    color = "orange",
                    hatch = patterns[0]
                    )
        else:
            plt.bar(
                    index + app_idx * bar_width, 
                    others_ratios,
                    bar_width,
                    #label = "others",
                    edgecolor = "black",
                    color = "tomato",
                    hatch = patterns[3]
                    )
            plt.bar(
                    index + app_idx * bar_width, 
                    optimization_ratios,
                    bar_width,
                    #label = "optimization",
                    edgecolor = "black",
                    color = "palegreen",
                    hatch = patterns[3]
                    )
            plt.bar(
                    index + app_idx * bar_width, 
                    comm_ratios,
                    bar_width,
                    #label = "communication",
                    edgecolor = "black",
                    color = "fuchsia",
                    hatch = patterns[2]
                    )
            plt.bar(
                    index + app_idx * bar_width, 
                    bubble_ratios,
                    bar_width,
                    #label = "bubble",
                    edgecolor = "black",
                    color = "aquamarine",
                    hatch = patterns[1]
                    )
            plt.bar(
                    index + app_idx * bar_width, 
                    compute_ratios,
                    bar_width,
                    #label = "compute",
                    edgecolor = "black",
                    color = "orange",
                    hatch = patterns[0]
                    )

    avg_comp_ratio /= (len(graphs) * len(models))
    avg_comm_ratio /= (len(graphs) * len(models))
    avg_bubble_ratio /= (len(graphs) * len(models))
    print("Average Computation Ratio: %.4f" % (avg_comp_ratio))
    print("Average Communication Ratio: %.4f" % (avg_comm_ratio))
    print("Average Bubble Ratio: %.4f" % (avg_bubble_ratio))

    plt.xlim([-0.25, 3.9])
    plt.xticks(index + bar_width, ("", "", ""))
    if show_legend:
        plt.legend()
    plt.savefig("%s_breakdown.pdf" % (method))
    plt.show()

if __name__ == "__main__":
    plot_breakdown("graph")
    plot_breakdown("pipeline", False)
    plot_breakdown("hybrid", False)


