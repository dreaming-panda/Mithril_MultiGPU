import os
import sys
import random
import statistics

datasets = [
        #"ogbn_products",
        "ogbn_arxiv",
        #"reddit",
        ]
num_gpus = {
        "ogbn_products": [3, 6],
        "ogbn_arxiv": [4, 8],
        "reddit": [3, 6]
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
            if "------------------------node id 0" in line:
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
                result_file = "./results/%s/result_%s.txt" % (
                        dataset, run
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

            print("Graph %s, Num GPU %s, Test Acc %.2f (+- %.2f), Runtime %.4f (+- %.4f)" % (
                dataset, gpus, avg_acc * 100., acc_stddev * 100., avg_runtime, runtime_stddev
                ))


