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

def read_async_gpu_accc(num_epoch, graph, model, res_dir, num_startup_epoches = 0):
    acc = []
    epoch_id = 0
    #print("./async/%s_%s.txt" % (graph, model))
    with open("./%s/%s_%s.txt" % (res_dir, graph, model), "r") as f:
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
                print(epoch_id, line)
                if epoch_id >= 10 * num_startup_epoches or (epoch_id + 1) % 10 == 0:
                    acc.append(float(line[-1]))
                epoch_id += 1
    print(len(acc))
    assert(len(acc) == num_epoch)
    return acc

if __name__ == "__main__":

    graph = "arxiv"
    model = "gcn"
    num_epoch = 500

    acc_random = read_async_gpu_accc(num_epoch, graph, model, "with_random_schedule")
    acc_no_random = read_async_gpu_accc(num_epoch, graph, model, "without_random_schedule")
    epoches = [i for i in range(num_epoch)]

    plt.plot(epoches, acc_random, label = "random")
    plt.plot(epoches, acc_no_random, label = "no_random")
    plt.legend()

    plt.show()
