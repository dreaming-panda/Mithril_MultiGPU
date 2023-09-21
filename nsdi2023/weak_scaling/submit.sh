#!/bin/bash
#SBATCH --output=./nsdi2023/weak_scaling/progress.txt
#SBATCH --partition=gpu
#SBATCH --gpus=16
#SBATCH --gpus-per-task=4
#SBATCH --exclusive

hostfile=./nsdi2023/weak_scaling/hostfile
mpirun nvidia-smi
mpirun ./gen_hostfile.sh > $hostfile

cd build
make -j
cd ..

dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/graph_parallel_datasets

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

for num_gpus in 16
do
    echo "Number of GPUs is $num_gpus"
    num_layers=$((num_gpus*8))
    chunks=$num_gpus
    num_dp_ways=$num_gpus

    for graph in reddit flickr physics
    do
        mkdir -p ./nsdi2023/weak_scaling/$graph
        echo "    Running $graph"
        mpirun -n $num_gpus --map-by node:PE=4 --hostfile $hostfile ./build/applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression --multi_label $multi_label > ./nsdi2023/weak_scaling/$graph/$num_gpus.txt 2>&1
    done
done



