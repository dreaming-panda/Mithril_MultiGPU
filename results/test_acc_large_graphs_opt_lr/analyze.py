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

plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2

def read_single_gpu_acc(num_epoch, graph, model):
    acc = []
    with open("./sync/%s_%s.txt" % (graph, model), "r") as f:
        while len(acc) < num_epoch:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            line = line.strip()
            if "Epoch" in line:
                #print(line)
                line = line.split(" ")
                acc.append(
                        float(line[-1])
                        )
    #print(len(acc))
    assert(len(acc) == num_epoch)
    return acc

def read_multi_gpu_accc(num_epoch, graph, model):
    acc = []
    with open("./async/%s_%s.txt" % (graph, model), "r") as f:
        while len(acc) < num_epoch:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if "********* Epoch" in line:
                while "++++++++++ Test Accuracy:" not in line:
                    line = f.readline()
                    if line == None or len(line) == 0:
                        assert(False)
                #print("'%s'" % (line))
                line = line.strip().split(" ")
                acc.append(float(line[-1]))
    assert(len(acc) == num_epoch)
    return acc

if __name__ == "__main__":
    num_epoch = 1000

    model = "gcn"

    for graph in ["reddit", "products", "arxiv"]:
        epoches = [i for i in range(num_epoch)]
        single_gpu_acc = read_single_gpu_acc(num_epoch, graph, model)
        multi_gpu_acc = read_multi_gpu_accc(num_epoch, graph, model)

        #print(epoches)
        #print(single_gpu_acc)

        plt.plot(epoches, single_gpu_acc, "-", label = "single-gpu")
        plt.plot(epoches, multi_gpu_acc, "-", label = "async")
        plt.legend()
        plt.ylabel("TestAcc")
        plt.xlabel("Epoch")
        plt.title("%s-%s" % (graph, model))
        plt.savefig("%s-%s.pdf" % (graph, model))
        plt.show()


