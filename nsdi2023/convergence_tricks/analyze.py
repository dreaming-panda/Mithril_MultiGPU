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

def extract_training_curve(result_file):
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
    return x, y

def plot_training_curve(result_file, curve_name, line = "-"):
    x, y = extract_training_curve(result_file)
    plt.plot(x, y, line, label = curve_name)

if __name__ == "__main__":
    for graph in graphs:
        fig, ax = plt.subplots(figsize=(5, 3))
        plot_training_curve(
                "./results/no_tricks/%s/gcnii/1.txt" % (graph),
                "No Tricks",
                line = ":"
                )
        plot_training_curve(
                "./results/no_trick_1/%s/gcnii/1.txt" % (graph),
                "No Trick 1",
                line = "--"
                )
        plot_training_curve(
                "./results/no_trick_2/%s/gcnii/1.txt" % (graph),
                "No Trick 2",
                line = "--+"
                )
        plot_training_curve(
                "./results/no_trick_3/%s/gcnii/1.txt" % (graph),
                "No Trick 3",
                line = "-."
                )
        plot_training_curve(
                "../overall_performance/results/pipeline/%s/gcnii/1.txt" % (graph),
                "With All Tricks",
                line = "-"
                )
        if graph == "reddit":
            plt.ylim([0.9, 0.96])
        plt.title(graph)
        if graph == "squirrel":
            plt.legend()
        plt.savefig("%s.pdf" % (graph))
        plt.show()




