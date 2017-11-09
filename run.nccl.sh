#!/usr/bin/env bash

MPI_EXE=/home/yuduo/tools/openmpi-3.0.0/bin/mpirun

$MPI_EXE -np 4 -hostfile hosts -npernode 2 -mca btl_openib_want_cuda_gdr 1 --mca btl_tcp_if_include ib0 /home/yuduo/work/mpi-gpu-direct-test/build/nccl_one_device_per_process
