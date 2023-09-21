#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=8
#SBATCH --gpus-per-task=4
#SBATCH --exclusive

hostfile=./nsdi2023/various_model_depth/hostfile
mpirun nvidia-smi
mpirun ./gen_hostfile.sh > $hostfile

cd build
make -j
cd ..

graph_dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/graph_parallel_datasets
pipeline_dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/pipeline_parallel_datasets

num_gpus=8
model=gcnii
num_layers=32
lr=1e-3
epoch=50
decay=0
dropout=0.5
eval_freq=-1
enable_compression=0
multi_label=0
exact_inference=1
seed=1

graph=squirrel
hunits=1000

echo "Running on $squirrel dataset"

for num_layers 8 16 32 64 128
do
    echo "Number of layers: $num_layers"

    # graph parallel
    chunks=$num_gpus
    num_dp_ways=$num_gpus
    result_dir=./nsdi2023/various_model_depth/graph/$graph
    mkdir -p $result_dir

    mpirun -n $num_gpus --map-by node:PE=4 --hostfile $hostfile ./build/applications/async_multi_gpus/$model --graph $graph_dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression --multi_label $multi_label > $result_dir/$num_layers.txt 2>&1
    
    # pipeline parallel
    chunks=$((num_gpus*4))
    num_dp_ways=1
    result_dir=./nsdi2023/various_model_depth/pipeline/$model
    mkdir -p $result_dir
    mpirun -n $num_gpus --map-by node:PE=4 --hostfile $hostfile ./build/applications/async_multi_gpus/$model --graph $pipeline_dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression --multi_label $multi_label > $result_dir/$num_layers.txt 2>&1
done


