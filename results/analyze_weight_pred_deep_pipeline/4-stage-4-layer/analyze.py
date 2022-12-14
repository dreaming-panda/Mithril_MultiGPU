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

def read_sync_acc(num_epoch, graph, model):
    acc = []
    with open("./sync/%s_%s.txt" % (model, graph), "r") as f:
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

def read_async_no_pred_accc(num_epoch, graph, model):
    acc = []
    with open("./async_no_pred/%s_%s.txt" % (model, graph), "r") as f:
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

def read_async_pred_accc(num_epoch, graph, model):
    acc = []
    with open("./async_pred/%s_%s.txt" % (model, graph), "r") as f:
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

if __name__ == "__main__":
    num_epoch = 5000

    model = "gcn"

    for graph in ["reddit", "products", "arxiv"]:
    #for graph in ["reddit", "arxiv"]:
        epoches = [i for i in range(num_epoch)]
        sync_acc = read_sync_acc(num_epoch, graph, model)
        async_no_pred_acc = read_async_no_pred_accc(num_epoch, graph, model)
        #async_matching_acc = read_async_matching_accc(num_epoch, graph, model)
        async_pred = read_async_pred_accc(num_epoch, graph, model)

        #print(epoches)
        #print(single_gpu_acc)

        plt.plot(epoches, sync_acc, "-", label = "sync")
        plt.plot(epoches, async_no_pred_acc, "-", label = "async_no_pred")
        #plt.plot(epoches, async_matching_acc, "-", label = "async_matching")
        plt.plot(epoches, async_pred, "-", label = "async_pred")

        plt.legend()
        plt.ylabel("TestAcc")
        plt.xlabel("Epoch")
        plt.title("%s-%s" % (graph, model))
        plt.savefig("%s-%s.pdf" % (graph, model))
        max_acc = max(sync_acc)
        plt.ylim([max_acc - 0.1, max_acc + 0.05])
        #plt.xlim([4000, 5000])
        plt.show()


