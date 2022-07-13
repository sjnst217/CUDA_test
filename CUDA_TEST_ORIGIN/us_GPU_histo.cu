
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
	unsigned char* buffer = (unsigned char*)big_random_block(SIZE); // 100MB�� �������� �����͸� ����


	//���� ����
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//test

	//GPU�޸� ����
	unsigned char* dev_buffer; //buffer�� ���� GPU�޸� �Ҵ�
	unsigned int* dev_histo;  //histo�� ���� GPU�޸� �Ҵ�
	cudaMalloc((void**)&dev_buffer, SIZE);
	cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_histo, 256 * sizeof(long));
	cudaMemset(dev_histo, 0, 256 * sizeof(int)); //�տ��� �ߴ� �� ó�� �� ���� 0���� �ʱ�ȭ

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int blocks = prop.multiProcessorCount;
	histo_kernel << <blocks * 2, 256 >> > (dev_buffer, SIZE, dev_histo);

	unsigned int histo[256];// ������ 8��Ʈ�� 256�� �������� � �ϳ��� ��(0x00~0xff)�� �� �� �����Ƿ� 
	//������׷��� �����Ϳ��� �� ���� ������ Ƚ���� ����ϱ� ���� 256���� ����Ұ� �ʿ�.

	cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	//��������
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %3.lf ms\n", elapsedTime);

	//��� ����ҿ� ���� ������׷� �հ谡 �츮�� ������ ������ Ȯ�����ش�.
	long histoCount = 0;
	for (int i = 0; i < 256; i++)
	{
		histoCount += histo[i];
	}
	printf("Histogram Sum: %ld\n", histoCount);
	

	//CPU�� �̿��ؼ� ������ Ƚ������ Ȯ��
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