#!/bin/bash
#SBATCH --output=./nsdi2023/shallow_models/progress.txt
#SBATCH --partition=gpu
#SBATCH --gpus=8
#SBATCH --gpus-per-task=4
#SBATCH --exclusive

hostfile=./nsdi2023/shallow_models/hostfile
echo "Configurating the hostfile"
mpirun nvidia-smi
mpirun ./gen_hostfile.sh > $hostfile

# compile the project
echo "Build the project"
cd build
make -j
cd ..

# run the experiments
echo "Run the experiments"

dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/graph_parallel_datasets

num_gpus=8
num_layers=4 # a shallow model that pipeline parallelism cannot support
model=gcn
hunits=100
lr=1e-3
epoch=50
decay=0
dropout=0.5
eval_freq=-1
enable_compression=0
multi_label=0
exact_inference=1
seed=1

for graph in flickr reddit
do
    for model in gcn graphsage
    do
        echo "Running $model on $graph"
        result_dir=./nsdi2023/shallow_models/$graph/$model
        mkdir -p $result_dir

        # run the graph parallel
        chunks=$num_gpus
        num_dp_ways=$num_gpus
        echo "    Running graph parallel"
        mpirun -n $num_gpus --map-by node:PE=4 --hostfile $hostfile ./build/applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression --multi_label $multi_label > $result_dir/graph.txt 2>&1

        chunks=$((num_gpus*4))
        num_dp_ways=4 # 2-stage
        echo "    Running hybrid parallel"
        mpirun -n $num_gpus --map-by node:PE=4 --hostfile $hostfile ./build/applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression --multi_label $multi_label > $result_dir/hybrid.txt 2>&1
    done
done






