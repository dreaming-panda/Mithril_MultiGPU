import os
import sys
import random
import statistics

datasets = [
        #"reddit",
        "ogbn_products",
        "ogbn_arxiv"
        ]
num_gpus = {
        "ogbn_products": [3],
        "ogbn_arxiv": [4, 8]
        }
num_runs = 5

def get_test_acc(result_file):
    with open(result_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if "achieved the highest validation accuracy" in line:
                line = line.strip().split(" ")
                return float(line[-1][:-1])

def get_runtime(result_file):
    with open(result_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if "Average per-epoch runtime:" in line:
                line = line.strip().split(" ")
                return float(line[-2])

if __name__ == "__main__":

    for dataset in datasets:
        for gpus in num_gpus[dataset]:
            avg_acc = 0
            accs = []
            avg_runtime = 0
            runtimes = []
            for run in range(num_runs):
                result_file = "./results/%s/%s_gpu/result_%s.txt" % (
                        dataset, gpus, run
                        )
                acc = get_test_acc(result_file)
                avg_acc += acc
                accs.append(acc)
                runtime = get_runtime(result_file)
                avg_runtime += runtime
                runtimes.append(runtime)

            avg_acc /= num_runs
            acc_stddev = statistics.stdev(accs)
            avg_runtime /= num_runs
            runtime_stddev = statistics.stdev(runtimes)

            print("Graph %s, Num GPU %s, Test Acc %.4f (+- %.4f), Runtime %.4f (+- %.4f)" % (
                dataset, gpus, avg_acc, acc_stddev, avg_runtime, runtime_stddev
                ))


