#!/bin/bash

hostname
nvidia-smi

cd build
make -j

num_layers=4
hunits=64
lr=1e-2
graph=cora
epoch=1500
decay=5e-4
chunks=1
dropout=0.0
seed=5
model=graphsage
eval_freq=1
exact_inference=0

mpirun -n 1 -N 1 ./applications/async_multi_gpus/$model --graph /home/amadeus/ssd512/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference

#model=graphsage
#mpirun -n 4 -N 4 ./applications/async_multi_gpus/$model --graph /home/amadeus/ssd512/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq 
#
#cd /home/amadeus/ssd512/baseline/Mithril_MultiGPU/build
#./applications/single_gpu/gcn_inference --graph /home/amadeus/ssd512/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --weight_file /tmp/saved_weights_pipe
