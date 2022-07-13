
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define imin(a, b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256; //threadsPerBlock�� ����

//�ſ� ���� N ���� ������ ���ҵ��� ���� �� ������ ���ؼ��� ���� N���� ������ ���� �ʿ��ѵ�, 
//�� ��� N �̻��� ���� ���� threadsPerBlock ����� �ʿ��ϴ�.
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);
//���� 32�� (N + threadsPerBlock - 1) / threadsPerBlock �� ���� ���� ����� �ߵ����Ѿ� �Ѵ�.

__global__ void dot(float* a, float* b, float* c)
{
	// ���⿡���� cache�� �� block�� thread�� �Ѱ����� �׸��� ���� ��ŭ�� ����� �޸𸮸� �Ҵ�
	//�� ��ϸ��� �� �����޸��� ���纻�� ���������� ���� ������ ��� �ε����� ������ �ʿ� X
	__shared__ float cache[threadsPerBlock];  //(������ cache) -> ���� �޸𸮿� CPU ���̿��� ������ ������ ���ؼ� �̸� �����͵��� �����صδ� ���� ���� �޸�
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid]; // a�� b�� �� ����
		tid += blockDim.x * gridDim.x; //(�������� ���� * ����� ����)�� ������
		//printf("\n%d\n", threadIdx.x); // while ���ο��� ��� �ϴ��� �ñ��ؼ�
	}

	//�� �������� �ջ� ����� �����ϴ� �� ����� ����
	cache[cacheIndex] = temp;  // �ϳ��� Ư�� �׸��� ��ȿ�� �����͸� ������ �ִ��� ������� �ʰ� �迭 ��ü�� �͸������� ���� �� �ֵ��� �����޸� ���۸� 
	
	//�ϵ��� � �������� ���� ��ɾ �����ϱ� �� ��� �� ��� �����尡 __syncthreads() ������ ��ɵ��� �ϼ��ϵ��� ����
	__syncthreads();  // �� ����� ��������� ����ȭ ����


	//reduction �ڵ�(�� �ڵ� ������ �������� ���ؼ��� threadsperblock�� 2�� �������̾�� �Ѵ�)
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
			//__syncthreads(); //�̷��� �ְ� �Ǹ� ������ �߻� ->120p �� �ڼ��� ������ ��������
		__syncthreads(); //cache�� ��� �����尡 ���� ����ϰ� ���� ���� �ݺ��� ����
		i /= 2;
	}

	if (cacheIndex == 0)
	{
		c[blockIdx.x] = cache[0];
	}
}

void test1()
{

	//double d = 12.345345; printf("%.4g\n\n\n", d); //%.6g -> ���ڰ� 6���� �Ǽ��� ��Ÿ����(12.35) (���࿡ %.4g �� ���ڰ� 4���� �Ǽ��� ��Ÿ���� (12.345))

	float* a, * b, c = 0, * partial_c;
	float* dev_a, * dev_b, * dev_c, * dev_partial_c;

	//a = new float[N]; // c++ ���� malloc�� ������ 
	a = (float*)malloc(sizeof(float) * N); //�� ����
	/*b = new float[N];
	partial_c = new float[blocksPerGrid];*/
	b = (float*)malloc(sizeof(float) * N); 
	partial_c = (float*)malloc(sizeof(int) * blocksPerGrid);
	

	cudaMalloc((void**)&dev_a, N * sizeof(float));
	cudaMalloc((void**)&dev_b, N * sizeof(float));
	cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));

	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	dot<<<blocksPerGrid, threadsPerBlock >>>(dev_a, dev_b, dev_partial_c);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Time to generate: %3.lf ms\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < blocksPerGrid; i++)
	{
		c += partial_c[i];
		printf("%g ", partial_c[i]);      //  .����g �� g �� ���̸� �˾ƺ��� ����
		printf(" %.6g\n", partial_c[i]);
	}
	#define sum_squares(x) (x*(x+1)*(2*x+1)/6) // 1���� x���� �� ������ ��
	printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);

	free(a);
	free(b);
	free(partial_c);
}

int main()
{
	test1();

	return 0;
}


//int myVar = 0;
//if(threadIdx.x % 2)
//	myVar = threadIdx.x;
//
//������ 3��° �ٿ� �������� �� ¦�� ���� ��������� if ������ �������� �ʱ� ������ 
//���� Ȧ�� ���� ������鸸�� �ش� ������ ������ ����.
//�̸� �����ϴ� ���� ¦������ ��������� �ƹ��͵� ���� ����. but ��� ������ ������ -> �׷��� ������


//�ٽ� ������ �غ��� GPU���� �б� ( if�� )�� ���� ( ���� ) ������ ���ǹ��� �����ϴ� �� ��ü�� ���� ���� �ƴ϶�
//GPU�� Warp���� ��������� �б������� �� ���� �б� ��θ� �����ؾ��ϱ� �����̴�.


