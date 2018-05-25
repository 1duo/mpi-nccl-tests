mkdir build
cd build
cmake -DNCCL_LIBRARY=/home/yuduo/tools/nccl-2.1.2/lib/libnccl.so -DNCCL_INCLUDE_DIR=/home/yuduo/tools/nccl-2.1.2/include/ ..
make -j
