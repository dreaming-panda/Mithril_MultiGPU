#!/bin/bash

hostname
nvidia-smi

cd build
make -j

num_layers=2
hunits=64
lr=3e-3
graph=ogbn_arxiv
epoch=100
decay=1e-5
chunks=1
dropout=0.5
seed=5
model=gcn
eval_freq=10

mpirun -n 1 -N 1 ./applications/async_multi_gpus/$model --graph /home/amadeus/ssd512/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq 

#model=gcnii
#mpirun -n 1 -N 1 ./applications/async_multi_gpus/$model --graph /home/amadeus/ssd512/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq 

#cd /home/amadeus/ssd512/baseline/Mithril_MultiGPU/build
#./applications/single_gpu/gcn_inference --graph /home/amadeus/ssd512/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --weight_file /tmp/saved_weights_pipe
