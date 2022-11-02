#!/bin/bash

#SBATCH -J Mithril           # Job name
#SBATCH -o mithril.o%j       # Name of stdout output file
#SBATCH -e mithril.e%j       # Name of stderr error file
#SBATCH -p gtx      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes 
#SBATCH --ntasks-per-node 1            # Per node # mpi tasks
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)

# please execute this batch on the top-level directory

mkdir -p build
bash prepare.sh
module load boost/1.66
cd build
cmake ..
make -j
cd ..

module list
pwd
date
# Set thread count (default value is 1)...

export OMP_NUM_THREADS=12

# Launch MPI code... 

#for dataset in cora citeseer pubmed ogbn-arxiv
for dataset in ogbn-arxiv
do
    ibrun ./build/applications/single_node/gcn --graph /work/03924/xuehaiq/maverick2/jingji/gnn_datasets/$dataset --layers 3 --hunits 128 --epoch 10000 --lr 0.001 --decay 0 | tee ./results/test_acc/single_gpu/${dataset}_gcn.txt
done

