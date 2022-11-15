#!/bin/bash
#----------------------------------------------------
# Example Slurm job script
# for TACC Stampede2 SKX nodes
#
#   *** Hybrid Job on SKX Normal Queue ***
# 
#       This sample script specifies:
#         10 nodes (capital N)
#         40 total MPI tasks (lower case n); this is 4 tasks/node
#         12 OpenMP threads per MPI task (48 threads per node)
#
# Last revised: 20 Oct 2017
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch skx.mpi.slurm" on Stampede2 login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do not use mpirun or mpiexec.
#
#   -- In most cases it's best to keep
#      ( MPI ranks per node ) x ( threads per rank )
#      to a number no more than 48 (total cores).
#
#   -- If you're running out of memory, try running
#      fewer tasks and/or threads per node to give each 
#      process access to more memory.
#
#   -- IMPI and MVAPICH2 both do sensible process pinning by default.
#
#----------------------------------------------------

#SBATCH -J Mithril           # Job name
#SBATCH -o mithril.o%j       # Name of stdout output file
#SBATCH -e mithril.e%j       # Name of stderr error file
#SBATCH -p gtx      # Queue (partition) name
#SBATCH -N 4             # Total # of nodes 
#SBATCH --ntasks-per-node 4            # Per node # mpi tasks
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)

# Other commands must follow all #SBATCH directives...

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

#ibrun ./build/tests/test_cuda_pipeline_parallel         # Use ibrun instead of mpirun or mpiexec
ibrun ./build/applications/async_multi_gpus/gcn --graph /work/03924/xuehaiq/maverick2/jingji/gnn_datasets/with_splitting/Citeseer --layers 2 --hunits 128 --epoch 500 --lr 1e-2 --decay 5e-4 --part hybrid

# ---------------------------------------------------
