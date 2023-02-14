import os
import sys
import random
import statistics

datasets = [
        "reddit",
        "ogbn_products"
        ]
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

if __name__ == "__main__":

    for dataset in datasets:
        avg_acc = 0
        accs = []
        for run in range(num_runs):
            result_file = "./results/%s/result_%s.txt" % (
                    dataset, run
                    )
            acc = get_test_acc(result_file)
            avg_acc += acc
            accs.append(acc)
        avg_acc /= num_runs
        stddev = statistics.stdev(accs)
        print("Graph %s, Test Acc %.4f (+- %.4f)" % (
            dataset, avg_acc, stddev
            ))
