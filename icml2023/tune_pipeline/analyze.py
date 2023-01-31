import matplotlib.pyplot as plt

scaledowns = [
        0.0, 0.1, 0.5, 1.0
        ]
chunks = [
        4, 8, 16, 32, 64
        ]

def get_epoch_time(result_file):
    with open(result_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                assert(False)
            if "per-epoch time:" in line:
                line = line.strip().split()
                return float(line[-2])

def get_test_Acc(result_file):
    epoch = []
    accs = []
    with open(result_file, "r") as f:
        line = f.readline()
        if line == None or len(line) == 0:
            break
        if "Version" in line and "TestAcc" in line:
            line = line.strip().split()
            accs.append(float(line[-1]))
            epoch.append(len(accs) * 10)
    return epoch, accs

if __name__ == "__main__":

    seed = 1234

    # analyze the scaling down factor
    for chunk in chunks:
        for scaledown in scaledowns:
            result_file = "%s/%s/%s/result.txt" % (
                    seed, scaledown, chunk
                    )
            epoch_time = get_epoch_time(result_file)
            x, y = get_test_Acc(result_file)
            for i in range(len(x)):
                x[i] *= epoch_time
            plt.plot(x, y, label = "scaledown = %s" % (scaledown))
        plt.legend()
        plt.xlabel("Training time")
        plt.ylabel("Test accuracy")
        plt.title("Chunks = %s" % (chunk))
        plt.show()





