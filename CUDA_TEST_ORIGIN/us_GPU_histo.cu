
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE (100 * 1024 * 1024)

__global__ void histo_kernel(unsigned char* buffer, long size, unsigned int* histo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (i < size)
	{
		atomicAdd(&(histo[buffer[i]]), 1);
		i += stride;
	}
}

int main()
{
	unsigned char* buffer = (unsigned char*)big_random_block(SIZE); // 100MB의 무작위의 데이터를 생성


	//성능 측정
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//test

	//GPU메모리 설정
	unsigned char* dev_buffer; //buffer에 대한 GPU메모리 할당
	unsigned int* dev_histo;  //histo에 대한 GPU메모리 할당
	cudaMalloc((void**)&dev_buffer, SIZE);
	cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_histo, 256 * sizeof(long));
	cudaMemset(dev_histo, 0, 256 * sizeof(int)); //앞에서 했던 것 처럼 각 값을 0으로 초기화

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int blocks = prop.multiProcessorCount;
	histo_kernel << <blocks * 2, 256 >> > (dev_buffer, SIZE, dev_histo);

	unsigned int histo[256];// 각각의 8비트는 256중 무작위로 어떤 하나의 값(0x00~0xff)이 될 수 있으므로 
	//히스토그램은 데이터에서 각 값이 등장한 횟수를 기록하기 위해 256개의 저장소가 필요.

	cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	//성능측정
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %3.lf ms\n", elapsedTime);

	//모든 저장소에 대한 히스토그램 합계가 우리가 예상한 값인지 확인해준다.
	long histoCount = 0;
	for (int i = 0; i < 256; i++)
	{
		histoCount += histo[i];
	}
	printf("Histogram Sum: %ld\n", histoCount);
	

	//CPU를 이용해서 동일한 횟수인지 확인
	for (int i = 0; i < SIZE; i++)
	{
		histo[buffer[i]]--;
		for (int i = 0; i < 256; i++)
		{
			if (histo[i] != 0)
			{
				printf("Failure at %d!\n", i);
			}
		}
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(dev_histo);
	cudaFree(dev_buffer);
	free(buffer);
	return 0;
}