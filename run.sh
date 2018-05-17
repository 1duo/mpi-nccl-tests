#!/usr/bin/env bash

MPI_EXE=mpirun


$MPI_EXE -np 2 -host 192.168.1.17,192.168.1.16 \
    -allow-run-as-root -mca btl_openib_warn_default_gid_prefix 0 \
    -mca btl_openib_want_cuda_gdr 1 --mca btl_tcp_if_include ib0 \
    build/nccl-two-communicators-diff-types-with-threads

exit


$MPI_EXE -np 2 -host 192.168.1.17,192.168.1.16 \
    -allow-run-as-root -mca btl_openib_warn_default_gid_prefix 0 \
    -mca btl_openib_want_cuda_gdr 1 --mca btl_tcp_if_include ib0 \
    build/nccl-two-streams-test-diff-types

exit


$MPI_EXE -np 2 -host 192.168.1.17,192.168.1.16 \
    -allow-run-as-root -mca btl_openib_warn_default_gid_prefix 0 \
    -mca btl_openib_want_cuda_gdr 1 --mca btl_tcp_if_include ib0 \
    build/nccl-two-streams-test


$MPI_EXE -np 2 -host 192.168.1.17,192.168.1.16 \
    -allow-run-as-root -mca btl_openib_warn_default_gid_prefix 0 \
    -mca btl_openib_want_cuda_gdr 1 --mca btl_tcp_if_include ib0 \
    build/nccl-two-streams-test-diff-streams
