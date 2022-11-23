import sys
import os

def get_valid_acc(result_file):
    with open(result_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                assert(False)
            line = line.strip()
            if "Highest validation acc:" in line:
                line = line.split(" ")
                acc = float(line[-1])
                return acc

def get_test_acc(result_file):
    with open(result_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                assert(False)
            line = line.strip()
            if "Target test acc:" in line:
                line = line.split(" ")
                acc = float(line[-1])
                return acc

num_layers = [2, 3]
#hunits = [16, 64, 256] FIXME
hunits = [16, 64]
learning_rates = ["0.001", "0.003", "0.01", "0.03"]
decays = ["0", "0.00001", "0.0001"]

if __name__ == "__main__":
    for model in ["gcn"]:
        for graph in ["reddit", "products", "arxiv"]:
            highest_valid_acc = 0
            optimal_parameter = {}
            target_test_acc = 0
            for num_layer in num_layers:
                for hunit in hunits:
                    for lr in learning_rates:
                        for decay in decays:
                            result_file = "./%s/%s/%s/%s/%s_%s.txt" % (
                                    num_layer, hunit, lr, decay,
                                    graph, model
                                    )
                            valid_acc = get_valid_acc(result_file)
                            test_acc = get_test_acc(result_file)
                            if valid_acc > highest_valid_acc:
                                highest_valid_acc = valid_acc
                                target_test_acc = test_acc
                                optimal_parameter = {
                                        "num_layer": num_layer,
                                        "hunit": hunit,
                                        "lr": lr,
                                        "decay": decay
                                        }
            print("Model = %s, graph = %s, highest valid accuracy = %.4f, corresponding test accuracy = %.4f, optimal parameters: %s" % (
                model, graph, highest_valid_acc, target_test_acc, optimal_parameter
                ))

