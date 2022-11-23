import os
import json
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import matplotlib as mpl
import math

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2

def get_error(data_0, data_1):
    assert(len(data_0) == len(data_1))
    error = 0
    for i in range(len(data_0)):
        error += (data_0[i] - data_1[i]) ** 2
    return error

def get_l2norm(data):
    norm = 0
    for i in range(len(data)):
        norm += data[i] ** 2
    return norm

if __name__ == "__main__":

    weights_each_epoch = []
    num_epoches = 0
    weights_this_epoch = []
    with open("./weights.txt", "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                weights_each_epoch.append(weights_this_epoch)
                break
            if "Epoch" in line:
                if num_epoches > 0:
                    weights_each_epoch.append(weights_this_epoch)
                #if num_epoches >= 100: # FIXME
                #    break
                weights_this_epoch = []
                print("Reading the weights of epoch %s" % (num_epoches))
                num_epoches += 1
                continue
            line = line.strip()
            line = line.split(" ")
            weights_this_epoch.append(float(line[-1]))

    print(len(weights_each_epoch), num_epoches)
    assert(len(weights_each_epoch) == num_epoches)
    num_weights = len(weights_each_epoch[0])
    for i in range(1, num_epoches):
        assert(len(weights_each_epoch[i]) == num_weights)

    error_without_prediction = []
    error_with_prediction = []
    error_with_prediction_2 = []
    epoches = [i for i in range(2, num_epoches)]
    dev_without_prediction = []
    dev_with_prediction = []

    for i in range(2, num_epoches):
        print("Calculating the prediction error of the weights when epoch = %s" % (i))

        #last_epoch_before_before_weights = weights_each_epoch[i - 3]
        last_epoch_before_weights = weights_each_epoch[i - 2]
        last_epoch_weights = weights_each_epoch[i - 1]
        this_epoch_weights = weights_each_epoch[i]

        l2norm = get_l2norm(this_epoch_weights)

        # default: use the previous-epoch weights to approximate current weights
        error_without_prediction.append(get_error(last_epoch_weights, this_epoch_weights))
        dev_without_prediction.append(math.sqrt(error_without_prediction[-1] / l2norm))

        # default: predict the current weights more accurately
        predicted_this_epoch_weights = []
        for j in range(num_weights):
            predicted_this_epoch_weights.append(
                    last_epoch_weights[j] + (last_epoch_weights[j] - last_epoch_before_weights[j])
                    )
        error_with_prediction.append(get_error(predicted_this_epoch_weights, this_epoch_weights))
        dev_with_prediction.append(math.sqrt(error_with_prediction[-1] / l2norm))

        ## another prediction method
        #predicted_this_epoch_weights = []
        #for j in range(num_weights):
        #    predicted_delta = (last_epoch_weights[j] - last_epoch_before_weights[j]) - (last_epoch_before_weights[j] - last_epoch_before_before_weights[j]) + (last_epoch_weights[j] - last_epoch_before_weights[j])
        #    predicted_this_epoch_weights.append(
        #            last_epoch_weights[j] + predicted_delta
        #            )
        #error_with_prediction_2.append(l2error(predicted_this_epoch_weights, this_epoch_weights))

    print(error_without_prediction)
    print(error_with_prediction)

    # plot the error figure
    plt.plot(epoches, dev_without_prediction, "-+", label = "without prediction")
    plt.plot(epoches, dev_with_prediction, "-+", label = "with prediction")
    #plt.ylim([0., 0.01])
    #plt.plot(epoches, error_with_prediction_2, "-+", label = "with prediction (v2)")
    #plt.yscale("log")
    plt.legend()
    plt.show()




