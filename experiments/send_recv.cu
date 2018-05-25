#include <iostream>
#include <mpi.h>

#define GPU_DIRECT_MPI

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);

  int mpiSize = 0;
  int mpiRank = 0;
  MPI_Status mpiStatus;

  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  if (mpiRank == 0) {
    std::cout << "Running on " << mpiSize << " Processes." << std::endl;
  }
  std::cout << "In Process [" << mpiRank << "/" << mpiSize << "]"
            << std::endl;

  const int numCommunications = 10;
  const int bufferSize = 256 * 1024 * 1024;  // 1GB data

  float *sendBuffer;
  float *recvBuffer;

  int numDevices = 0;
  cudaGetDeviceCount(&numDevices);
  cudaSetDevice(numDevices - 1);

#ifndef GPU_DIRECT_MPI
  float* cpuSendBuffer;
  float* cpuRecvBuffer;
#endif

  float *gpuSendBuffer;
  float *gpuRecvBuffer;

  cudaMalloc(&gpuSendBuffer, sizeof(float) * bufferSize);
  cudaMalloc(&gpuRecvBuffer, sizeof(float) * bufferSize);

  for (int i = 0; i < numCommunications; ++i) {

#ifndef GPU_DIRECT_MPI
    cpuSendBuffer = (float*) malloc(sizeof(float) * bufferSize);
    cpuRecvBuffer = (float*) malloc(sizeof(float) * bufferSize);
    cudaMemcpy(cpuSendBuffer, gpuSendBuffer, sizeof(float) * bufferSize,
            cudaMemcpyDeviceToHost);
    sendBuffer = cpuSendBuffer;
    recvBuffer = cpuRecvBuffer;
#else
    sendBuffer = gpuSendBuffer;
    recvBuffer = gpuRecvBuffer;
#endif

    int left, right;
    right = (mpiRank + 1) % mpiSize;
    left = mpiRank - 1;
    if (left < 0) {
      left = mpiSize - 1;
    }

    /*
     std::cout << "MPI_Sendrecv Arguments:" << std::endl
     << "left: " << left << " right: " << right
     << " sendBuffer: " << sendBuffer << " recvBuffer: "
     << recvBuffer << std::endl;
     */

    MPI_Sendrecv(sendBuffer, bufferSize, MPI_FLOAT, left, 123, recvBuffer,
                 bufferSize, MPI_FLOAT, right, 123, MPI_COMM_WORLD, &mpiStatus);

#ifndef GPU_DIRECT_MPI
    cudaMemcpy(cpuRecvBuffer, gpuRecvBuffer, sizeof(float) * bufferSize,
            cudaMemcpyDeviceToHost);
#endif
  }

  cudaFree(gpuSendBuffer);
  cudaFree(gpuRecvBuffer);

#ifndef GPU_DIRECT_MPI
  free(cpuSendBuffer);
  free(cpuRecvBuffer);
#endif

  MPI_Finalize();

  return 0;
}
