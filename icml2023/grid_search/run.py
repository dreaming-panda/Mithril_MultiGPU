import os
import sys
import time

# number of combinations:
# 4 x 2 s 4 x 3 = 96
learning_rates = [
        3e-4, 1e-3, 3e-3, 1e-2
        ]
decays = [
        0, 1e-5
        ]
hunits = [
        64, 128, 256, 512
        ]
dropouts = [
        0.3, 0.5, 0.7
        ]
graph_path = "$PROJECT/gnn_datasets/reordered"

def train_gcn(lr, decay, hunit, dropout, full_graph_path, graph, weight_file):
    result_dir = "./icml2023/grid_search/results/%s/%s/%s/%s/%s" % (
            graph, hunit, lr, decay, dropout
            )
    os.system("mkdir -p %s" % (result_dir))

    t = - time.time()
    result_file = result_dir + "/result.txt"
    command = "mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./build/applications/single_gpu/gcn --graph %s --layers 4 --hunits %s --epoch 3000 --lr %s --decay %s --dropout %s --weight_file %s > %s 2>&1" % (
            full_graph_path, hunit, lr, decay, dropout, weight_file, result_file
            )
    os.system(command)
    command = "mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./build/applications/single_gpu/gcn_inference --graph %s --layers 4 --hunits %s --weight_file %s >> %s 2>&1" % (
            full_graph_path, hunit, weight_file, result_file
            )
    os.system(command)
    t += time.time()
    print("    It takes %.3f s" % (t))


if __name__ == "__main__":

    graph = sys.argv[1]
    full_graph_path = graph_path + "/" + graph

    weight_file = "checkpointed_weights"
    if len(sys.argv) > 2:
        weight_file = sys.argv[2]

    print("Running the grid search on graph %s" % (
        graph
        ))
    for lr in learning_rates:
        for decay in decays:
            for hunit in hunits:
                for dropout in dropouts:
                    print("training with learning rate %.5f, decay %.5f, hunit %s, dropout rate %.5f" % (
                        lr, decay, hunit, dropout
                        ))
                    sys.stdout.flush()
                    train_gcn(
                            lr, decay, hunit, dropout, full_graph_path, graph, weight_file
                            );


