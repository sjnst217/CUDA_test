
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE (100 * 1024 * 1024)

__global__ void histo_kernel(unsigned char* buffer, long size, unsigned int* histo)
{
	__shared__ unsigned int temp[256]; //공유메모리 버퍼 할당
	temp[threadIdx.x] = 0; // 0으로 초기화
	__syncthreads(); // 모든 스레드의 기록 작업을 끝내고 다음 과정으로 넘어가도록 해줌

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	while (i < size)
	{
		atomicAdd(&temp[buffer[i]], 1); //전역 메모리 histo 대신 공유 메모리 temp를 사용
		i += offset;
	}
	__syncthreads();

	atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]); //각 블록의 임시 히스토그램(temp)들을 전역 버퍼(histo)로 합쳐줌
	//스레드에서의 개수는 어떠한 수로도 바뀔 수 있으므로, 최종 히스토그램을 위한 모든 블록의 히스토그램들의 합은
	//각 블록의 히스토그램 각각의 항목을 최종의 히스토그램의 각각의 항목에 합한것과 같다
	//그러므로 원자적으로 수행되어야 함
	//256개의 스레드를 사용하기로 했고, 256개의 히스토그램 저장소들(temp, histo)을 가졌으므로
	//각 스레드는 하나의 단일 저장소를 최종의 히스토그램에 원자적으로 더해줌 ->원자적 합산이 제공되므로 결과값은 항상 동일

}

int main()
{
	unsigned char* buffer = (unsigned char*)big_random_block(SIZE); // 100MB의 무작위의 데이터를 생성

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
//미완