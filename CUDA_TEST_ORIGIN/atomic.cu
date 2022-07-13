
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE (100 * 1024 * 1024)

__global__ void histo_kernel(unsigned char* buffer, long size, unsigned int* histo)
{
	__shared__ unsigned int temp[256];
	temp[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	while (i < size)
	{
		atomicAdd(&temp[buffer[i]], 1);
		i += offset;
	}

	__syncthreads();
	atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

int main()
{
	unsigned char* buffer = (unsigned char*)big_random_block(SIZE); // 100MB�� �������� �����͸� ����

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	unsigned char* dev_buffer;
	unsigned int* dev_histo;
	cudaMalloc((void**)&dev_buffer, SIZE);
	cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_histo, 256 * sizeof(long));
	cudaMemset(dev_histo, 0, 256 * sizeof(int));

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int blocks = prop.multiProcessorCount;
	histo_kernel << <blocks * 2, 256 >> > (dev_buffer, SIZE, dev_histo);

	unsigned int histo[256]; 

	cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %3.lf ms\n", elapsedTime);


	long histoCount = 0;
	for (int i = 0; i < 256; i++)
	{
		histoCount += histo[i];
	}
	printf("Histogram Sum: %ld\n", histoCount);

	for (int i = 0; i < SIZE; i++)
	{
		histo[buffer[i]]--;
	}
	for (int i = 0; i < 256; i++)
	{
		if (histo[i] != 0)
		{
			printf("Failure at %d!\n", i);
		}
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(dev_histo);
	cudaFree(dev_buffer);
	free(buffer);
	return 0;
}
//�̿�