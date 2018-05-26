//
// Created by novuforce on 5/25/18.
//

#include <cstdio>
#include <ctime>
#include <chrono>
#include "cuda_runtime.h"
#include "common.h"

int main(int argc, char *argv[]) {

    size_t packageSize = 256 * 1024 * 1024 / 8;

    auto cluster = new Cluster();
    cluster->Init();

    float *sendBuffGpu;
    float *recvBuffGpu;

    CUDA_CHECK(cudaMalloc(&sendBuffGpu, packageSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recvBuffGpu, packageSize * sizeof(float)));
    auto sendBuffCpu = (float *) malloc(packageSize * sizeof(float));

    CUDA_CHECK(cudaMemcpy(sendBuffGpu, sendBuffCpu, packageSize, cudaMemcpyHostToDevice));

    auto timer = new HighResolutionTimer();

    timer->Start();
    for (auto i = 0; i < 10; ++i) {
        cluster->AllReduceNCCL(sendBuffGpu, recvBuffGpu, packageSize);
    }
    timer->Stop();

    std::cout << "Elapsed: " << timer->Elapsed() << "s." << std::endl;

    free(sendBuffCpu);
    CUDA_CHECK(cudaFree(sendBuffGpu));
    CUDA_CHECK(cudaFree(recvBuffGpu));

    cluster->Finalize();

    return 0;
}
