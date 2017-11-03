#!/usr/bin/env bash

MPI_EXE=/home/yuduo/tools/openmpi-3.0.0/bin/mpirun

# InfiniBand with RDMA
time $MPI_EXE --tag-output -np 2 -map-by ppr:1:node --host 192.168.56.142,192.168.56.143 -mca btl_openib_want_cuda_gdr 1 --mca btl_tcp_if_include ib0 --mca orte_base_help_aggregate 0 --mca mpi_common_cuda_cumemcpy_async 1 -mca btl_openib_cuda_rdma_limit 65537000000 /home/yuduo/work/mpi-gpu-direct-test/build/reduce_scatter

# Ethernet
time $MPI_EXE --tag-output -np 2 -map-by ppr:1:node --host 172.27.10.142,172.27.10.143 -mca btl_openib_want_cuda_gdr 1 --mca btl_tcp_if_include enp129s0f0 --mca orte_base_help_aggregate 0 --mca mpi_common_cuda_cumemcpy_async 1 /home/yuduo/work/mpi-gpu-direct-test/build/send_recv

# InfiniBand without RDMA
time $MPI_EXE --tag-output -np 2 -map-by ppr:1:node --host 172.27.10.142,172.27.10.143 -mca btl_openib_want_cuda_gdr 0 --mca btl_tcp_if_include ib0 --mca orte_base_help_aggregate 0 --mca mpi_common_cuda_cumemcpy_async 1 /home/yuduo/work/mpi-gpu-direct-test/build/send_recv
