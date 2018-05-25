FILE(GLOB_RECURSE SOURCES "*.cu" "*.cpp" "*.c" "*.h")
FOREACH (source ${SOURCES})
    GET_FILENAME_COMPONENT(name ${source} NAME_WE)
    CUDA_ADD_EXECUTABLE(${name} ${source})
    TARGET_LINK_LIBRARIES(${name} /usr/local/cuda/lib64/libcudart.so
            ${MPI_LIBRARIES})
    TARGET_LINK_LIBRARIES(${name} ${NCCL_LIBRARY})
ENDFOREACH (source)
