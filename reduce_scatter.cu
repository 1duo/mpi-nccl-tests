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

	const int numCommunications = 1;
	const int bufferSize = 4 * 256 * 1024 * 1024;  // 4 * 1GB data
	// const int bufferSize = 25610269;

	float* sendBuffer;
	float* recvBuffer;

	int numDevices = 0;
	cudaGetDeviceCount(&numDevices);
	cudaSetDevice(numDevices - 1);

	float* gpuSendBuffer;
	float* gpuRecvBuffer;

	cudaMalloc(&gpuSendBuffer, sizeof(float) * bufferSize);
	cudaMalloc(&gpuRecvBuffer, sizeof(float) * bufferSize);
	cudaMemset(gpuSendBuffer, 1., bufferSize * sizeof(float));

#ifndef GPU_DIRECT_MPI
	float* cpuSendBuffer;
	float* cpuRecvBuffer;
#endif

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

		const int blockSize = bufferSize / mpiSize;
		int sendBufferSize = 0;
		int recvBufferSize = 0;
		int sendStartBlock = 0;
		int recvStartBlock = 0;

		int interval = mpiSize;
		while (interval > 1) {
			if (mpiRank % interval < interval / 2) {
				sendStartBlock = recvStartBlock + interval / 2;
				sendBufferSize =
						(sendStartBlock + interval / 2 == mpiSize) ?
								bufferSize - blockSize * sendStartBlock :
								blockSize * interval / 2;
				recvBufferSize =
						(recvStartBlock + interval / 2 == mpiSize) ?
								bufferSize - blockSize * recvStartBlock :
								blockSize * interval / 2;

				std::cout << "MPI_Sendrecv Arguments 1:" << std::endl
						<< "sendBuffer + blockSize * sendStartBlock: "
						<< sendBuffer + blockSize * sendStartBlock
						<< " sendBufferSize: " << sendBufferSize
						<< " recvBufferSize: " << recvBufferSize << std::endl
						<< " mpiRank + interval / 2: " << mpiRank + interval / 2
						<< " sendBuffer: " << sendBuffer << " recvBuffer: "
						<< recvBuffer << std::endl;

				MPI_Sendrecv(sendBuffer + blockSize * sendStartBlock,
						sendBufferSize, MPI_FLOAT, mpiRank + interval / 2,
						mpiRank, recvBuffer, recvBufferSize, MPI_FLOAT,
						mpiRank + interval / 2, mpiRank + interval / 2,
						MPI_COMM_WORLD, &mpiStatus);
			} else {
				sendStartBlock = recvStartBlock;
				recvStartBlock = recvStartBlock + interval / 2;
				sendBufferSize =
						(sendStartBlock + interval / 2 == mpiSize) ?
								bufferSize - blockSize * sendStartBlock :
								blockSize * interval / 2;
				recvBufferSize =
						(recvStartBlock + interval / 2 == mpiSize) ?
								bufferSize - blockSize * recvStartBlock :
								blockSize * interval / 2;

				std::cout << "MPI_Sendrecv Arguments 2:" << std::endl
						<< "sendBuffer + blockSize * sendStartBlock: "
						<< sendBuffer + blockSize * sendStartBlock
						<< " sendBufferSize: " << sendBufferSize
						<< " recvBufferSize: " << recvBufferSize << std::endl
						<< " mpiRank - interval / 2: " << mpiRank - interval / 2
						<< " sendBuffer: " << sendBuffer << " recvBuffer: "
						<< recvBuffer << std::endl;

				MPI_Sendrecv(sendBuffer + blockSize * sendStartBlock,
						sendBufferSize, MPI_FLOAT, mpiRank - interval / 2,
						mpiRank, recvBuffer, recvBufferSize, MPI_FLOAT,
						mpiRank - interval / 2, mpiRank - interval / 2,
						MPI_COMM_WORLD, &mpiStatus);
			}
			interval /= 2;
		}

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
