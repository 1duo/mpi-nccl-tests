//
// Created by novuforce on 5/25/18.
//

#ifndef MPI_GPU_DIRECT_TEST_COMMON_H
#define MPI_GPU_DIRECT_TEST_COMMON_H

#include <unistd.h>
#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

#define MPI_CHECK(cmd) do {                      \
  int e = cmd;                                   \
  if (e != MPI_SUCCESS) {                        \
    printf("Failed: MPI error %s:%d '%d'\n",     \
     __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                          \
  }                                              \
} while(0)

#define CUDA_CHECK(cmd) do {                     \
  cudaError_t e = cmd;                           \
  if (e != cudaSuccess) {                        \
    printf("Failed: CUDA error %s:%d '%s'\n",    \
     __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                          \
  }                                              \
} while(0)

#define NCCL_CHECK(cmd) do {                     \
  ncclResult_t r = cmd;                          \
  if (r != ncclSuccess) {                        \
    printf("Failed, NCCL error %s:%d '%s'\n",    \
     __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                          \
  }                                              \
} while(0)

/*
static uint64_t getHostHash(const char *string) {
    // based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void getHostName(char *hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}
*/

class Cluster {
public:
    void Init() {
        MPI_CHECK(MPI_Init(NULL, NULL));
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size_));
        std::cout << "Init " << rank_ << "/" << size_ << std::endl;
        InitNCCL();
    }

    void Finalize() {
        MPI_CHECK(MPI_Finalize());
        FinalizeNCCL();
    }

    int rank() {
        return rank_;
    }

    int size() {
        return size_;
    }

    void InitNCCL() {
        if (rank_ == 0) {
            ncclGetUniqueId(&nccl_id_);
        }
        MPI_CHECK(MPI_Bcast((void *) &nccl_id_, sizeof(nccl_id_), MPI_BYTE, 0, MPI_COMM_WORLD));
        CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
        NCCL_CHECK(ncclCommInitRank(&nccl_comm_, size_, nccl_id_, rank_));
    }

    void AllReduceNCCL(void *sendBuff, void *recvBuff, size_t packageSize) {
        NCCL_CHECK(ncclAllReduce((const void *) sendBuff, (void *) recvBuff, packageSize,
                                 ncclFloat, ncclSum, nccl_comm_, cuda_stream_));
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
    }

    void FinalizeNCCL() {
        NCCL_CHECK(ncclCommDestroy(nccl_comm_));
    }

private:
    int rank_;
    int size_;
    ncclUniqueId nccl_id_;
    ncclComm_t nccl_comm_;
    cudaStream_t cuda_stream_;
};

class HighResolutionTimer {
public:
    void Start() {
        t1_ = std::chrono::high_resolution_clock::now();
    }

    void Stop() {
        t2_ = std::chrono::high_resolution_clock::now();
    }

    double Elapsed() {
        time_span_ = std::chrono::duration_cast<std::chrono::duration<double>>(t2_ - t1_);
        return time_span_.count();
    }

private:
    std::chrono::high_resolution_clock::time_point t1_;
    std::chrono::high_resolution_clock::time_point t2_;
    std::chrono::duration<double> time_span_;
};

#endif //MPI_GPU_DIRECT_TEST_COMMON_H
