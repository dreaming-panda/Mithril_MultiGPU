#!/bin/bash

RESULT_DIR=$PWD

cd ../../build/applications/async_multi_gpus
make -j 

./gcn --graph /ssd512/gnn_datasets/with_split/reordered/ogbn_arxiv_bi --layers 3 --hunits 256 --lr 1e-2 --decay 0 --epoch 500 --part hybrid --startup 0 --random 1 | tee $RESULT_DIR/with_random_schedule/arxiv_gcn.txt
./gcn --graph /ssd512/gnn_datasets/with_split/reordered/ogbn_arxiv_bi --layers 3 --hunits 256 --lr 1e-2 --decay 0 --epoch 500 --part hybrid --startup 0 --random 0 | tee $RESULT_DIR/without_random_schedule/arxiv_gcn.txt

