#!/bin/bash

RESULT_DIR=$PWD

cd ../../../build/applications/async_multi_gpus
make -j 

for chunks in 1 2 8 32 128 512
do
    RES_DIR=$RESULT_DIR/$chunks
    mkdir -p $RES_DIR
    echo "Running with $chunks chunks"
    ./gcn --graph /ssd512/gnn_datasets/with_split/reordered/ogbn_arxiv_bi --layers 2 --hunits 256 --lr 1e-3 --decay 0 --epoch 2000 --chunks $chunks > $RES_DIR/arxiv_gcn.txt
done


