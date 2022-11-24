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

def read_sync_gpu_acc(num_epoch, graph, model):
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

def read_sync_gpu_valid_acc(num_epoch, graph, model):
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
                        float(line[-2].split("\t")[0])
                        )
    #print(len(acc))
    assert(len(acc) == num_epoch)
    return acc

def read_async_gpu_accc(num_epoch, graph, model, num_startup_epoches):
    acc = []
    epoch_id = 0
    #print("./async/%s_%s.txt" % (graph, model))
    with open("./async/%s/%s_%s.txt" % (num_startup_epoches, graph, model), "r") as f:
        #print(len(acc), num_epoch)
        while len(acc) < num_epoch:
            line = f.readline()
            #print("read = ", line)
            if line == None or len(line) == 0:
                break
            if "********* Epoch" in line:
                while "++++++++++ Test Accuracy:" not in line:
                    line = f.readline()
                    if line == None or len(line) == 0:
                        assert(False)
                #print("'%s'" % (line))
                line = line.strip().split(" ")
                if epoch_id >= 10 * num_startup_epoches or (epoch_id + 1) % 10 == 0:
                    acc.append(float(line[-1]))
                epoch_id += 1
    #print(len(acc))
    assert(len(acc) == num_epoch)
    return acc

if __name__ == "__main__":
    num_epoch = 3000

    model = "gcn"
    for graph in ["reddit", "products", "arxiv"]:
        print()
        epoches = [i for i in range(num_epoch)]
        sync_acc = read_sync_gpu_acc(num_epoch, graph, model)
        sync_valid_acc = read_sync_gpu_valid_acc(num_epoch, graph, model)
        plt.plot(epoches, sync_acc, "-", label = "sync")
        highest_acc = 0
        highest_acc_epoch = 0
        for i in range(len(sync_acc)):
            if sync_acc[i] > highest_acc:
                highest_acc = sync_acc[i]
                highest_acc_epoch = i + 1
        print("Model = %s, Graph = %s, it takes sync %s epoches to reach the target accuracy %.4f." % (
            model, graph, highest_acc_epoch + 1, highest_acc
            ))
        for num_startup_epoches in [0, 10, 20, 30, 40, 50, 100, 200, 300]:
            async_acc = read_async_gpu_accc(num_epoch, graph, model, num_startup_epoches)
            reached_same_acc = False
            for i in range(len(async_acc)):
                if async_acc[i] >= highest_acc:
                    reached_same_acc = True
                    print("Model = %s, Graph = %s, it takes async (startup = %s) %s epoches to reach the target accuracy." % (
                        model, graph, num_startup_epoches, i + 1
                        ))
                    break
            if not reached_same_acc:
                print("Model = %s, Graph = %s, async (startup = %s) cannot reach the target accuracy." % (
                    model, graph, num_startup_epoches
                    ))
            #plt.plot(epoches, async_acc, "-", label = "async (startup=%s)" % (num_startup_epoches))
        max_acc = max(sync_acc)
        plt.ylim([max_acc - 0.2, max_acc + 0.02])
        plt.legend()
        #plt.show()





