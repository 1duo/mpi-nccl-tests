mkdir build
cd build
cmake3 -DNCCL_LIBRARY=/usr/local/nccl_2.1.15-1+cuda9.1_x86_64/lib/libnccl.so -DNCCL_INCLUDE_DIR=/usr/local/nccl_2.1.15-1+cuda9.1_x86_64/include/ ..
