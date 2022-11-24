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

    for graph in ["reddit", "products", "arxiv"]:
        
        model = "gcn"

        num_epoch = 500
        epoches = [i for i in range(num_epoch)]

        #for chunks in [1, 2, 8, 32, 128, 512]:
        #    acc = read_async_gpu_accc(num_epoch, graph, model, "3-layer/%s" % (chunks))
        #    plt.plot(epoches, acc, "-+", label = "%s-chunks" % (chunks))

        #plt.xlabel("epoch")
        #plt.ylabel("test acc")
        #plt.title("ogbn-arxiv 3-layer GCN")
        #plt.legend()
        #plt.show()

        num_epoch = 2000
        epoches = [i for i in range(num_epoch)]

        #for chunks in [1, 2, 8, 32, 128, 512]:
        for chunks in [1, 2, 8, 32, 128]:
            acc = read_async_gpu_accc(num_epoch, graph, model, "2-layer/%s" % (chunks))
            plt.plot(epoches, acc, "-", label = "%s-chunks" % (chunks))
        plt.xlabel("epoch")
        plt.ylabel("test acc")
        plt.title("%s 2-layer GCN" % (graph))
        max_acc = max(acc)
        plt.ylim([max_acc - 0.005, max_acc + 0.002])
        plt.legend()
        plt.show()


