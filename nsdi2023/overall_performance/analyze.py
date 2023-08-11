import os

graphs = [
        #"squirrel",
        "flickr",
        "reddit"
        ]
models = [
        "gcn",
        "graphsage",
        "gcnii"
        ]
methods = [
        "graph",
        #"pipeline",
        #"hybrid"
        ]
num_runs = 3

def get_epoch_time(result_file):
    with open(result_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if "****** Epoch Time (Excluding Evaluation Cost):" in line:
                line = line.strip().split(" ")
                t = float(line[-3])
                return t
    assert(False)

def get_breakdown_time(result_file, breakdown_name):
    with open(result_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if ("Cluster-Wide Average, %s" % (breakdown_name)) in line:
                line = line.strip().split(" ")
                t = float(line[3])
                return t
    assert(False)

def get_communication_volume(result_file, comm_type):
    with open(result_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if ("%s communication (cluster-wide, per-epoch):" % (comm_type)) in line:
                line = line.strip().split(" ")
                t = float(line[-2])
                return t
    assert(False)

def get_test_accuracy(result_file):
    with open(result_filem, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if "Target test_acc: " in line:
                line = line.strip().split(" ")
                t = float(line[-1])
                return t
    assert(False)

def print_runtimes():
    print("Epoch Time (unit: s)")
    print(methods)
    for graph in graphs:
        for model in models:
            epoch_times = []
            for method in methods:
                s = 0.
                for seed in range(1, num_runs + 1):
                    result_file = "./results/%s/%s/%s/%s.txt" % (
                            method, graph, model, seed
                            )
                    s += get_epoch_time(result_file)
                s /= float(num_runs)
                epoch_times.append("%.2f" % (s))
            print(graph, model, epoch_times)

def print_communication_volume():
    print("Communication Volume (unit: GB)")
    print(methods)
    for graph in graphs:
        for model in models:
            epoch_times = []
            for method in methods:
                s = 0.
                for seed in range(1, num_runs + 1):
                    result_file = "./results/%s/%s/%s/%s.txt" % (
                            method, graph, model, seed
                            )
                    s += get_communication_volume(result_file, "Total")
                s /= float(num_runs)
                epoch_times.append("%.2f" % (s))
            print(graph, model, epoch_times)

if __name__ == "__main__":
    print_runtimes()
    print_communication_volume()



