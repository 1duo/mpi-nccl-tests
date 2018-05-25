#include <cstdio>
#include <ctime>
#include <chrono>
#include "cuda_runtime.h"
#include "common.h"

int main(int argc, char *argv[]) {

    auto cluster = new Cluster();
    cluster->Init();

    size_t packageSize = 256 * 1024 * 1024 / 8;

    CUDA_CHECK(cudaSetDevice(0));

    float *sendBuffGpu;
    float *recvBuffGpu;

    CUDA_CHECK(cudaMalloc(&sendBuffGpu, packageSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recvBuffGpu, packageSize * sizeof(float)));
    auto sendBuffCpu = (float *) malloc(packageSize * sizeof(float));
    CUDA_CHECK(cudaMemcpy(sendBuffGpu, sendBuffCpu, packageSize, cudaMemcpyHostToDevice));

    auto timer = new HighResolutionTimer();

    timer->Start();
    for (auto i = 0; i < 20; ++i) {
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
