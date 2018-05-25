mkdir build
cd build
cmake3 -DNCCL_LIBRARY=/opt/nccl_2.1.15-1+cuda9.1_x86_64/lib/libnccl.so \
    -DNCCL_INCLUDE_DIR=/opt/nccl_2.1.15-1+cuda9.1_x86_64/include/ ..
make -j
