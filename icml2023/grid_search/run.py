import os
import sys
import time

# products
# combinations: 3 x 1 x 3 x 5 = 45
learning_rates = [
        1e-3, 3e-4, 3e-3
        ]
decays = [
        0
        ]
hunits = [
        16, 32, 48
        ]
dropouts = [
        0.3, 0.4, 0.5, 0.6, 0.7
        ]

graph_path = "$PROJECT/gnn_datasets/reordered"

def train_gcn(lr, decay, hunit, dropout, full_graph_path, graph, weight_file):
    result_dir = "./icml2023/grid_search/results/%s/%s/%s/%s/%s" % (
            graph, hunit, lr, decay, dropout
            )
    os.system("mkdir -p %s" % (result_dir))

    t = - time.time()
    result_file = result_dir + "/result.txt"
    command = "mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./build/applications/single_gpu/gcn --graph %s --layers 8 --hunits %s --epoch 5000 --lr %s --decay %s --dropout %s --weight_file %s > %s 2>&1" % (
            full_graph_path, hunit, lr, decay, dropout, weight_file, result_file
            )
    os.system(command)
    command = "mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./build/applications/single_gpu/gcn_inference --graph %s --layers 8 --hunits %s --weight_file %s >> %s 2>&1" % (
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

    if len(sys.argv) > 3:
        learning_rates = [float(sys.argv[3])]

    if len(sys.argv) > 4:
        decays = [float(sys.argv[4])]

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


