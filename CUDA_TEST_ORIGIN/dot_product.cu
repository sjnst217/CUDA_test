
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define imin(a, b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256; //threadsPerBlock은 고정

//매우 작은 N 개의 데이터 원소들이 있을 때 내적을 위해서는 오직 N개의 스레드 만이 필요한데, 
//이 경우 N 이상의 가장 작은 threadsPerBlock 배수가 필요하다.
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);
//따라서 32나 (N + threadsPerBlock - 1) / threadsPerBlock 중 작은 수로 블록을 발동시켜야 한다.

__global__ void dot(float* a, float* b, float* c)
{
	// 여기에서의 cache는 각 block의 thread당 한개씩의 항목을 갖는 만큼의 충분한 메모리를 할당
	//각 블록마다 이 공유메모리의 복사본을 개별적으로 갖기 때문에 블록 인덱스를 포함할 필요 X
	__shared__ float cache[threadsPerBlock];  //(원래의 cache) -> 실제 메모리와 CPU 사이에서 빠르게 전달을 위해서 미리 데이터들을 저장해두는 좀더 빠른 메모리
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid]; // a와 b의 곱 저장
		tid += blockDim.x * gridDim.x; //(스레드의 갯수 * 블록의 갯수)를 더해줌
		//printf("\n%d\n", threadIdx.x); // while 내부에서 어떻게 하는지 궁금해서
	}

	//각 스레드의 합산 결과를 저장하는 데 사용할 것임
	cache[cacheIndex] = temp;  // 하나의 특정 항목이 유효한 데이터를 가지고 있는지 고려하지 않고 배열 전체를 맹목적으로 합할 수 있도록 공유메모리 버퍼를 
	
	//하드웨어가 어떤 스레드의 다음 명령어를 수행하기 전 블록 내 모든 스레드가 __syncthreads() 이전의 명령들을 완수하도록 해줌
	__syncthreads();  // 이 블록의 스레드들을 동기화 해줌


	//reduction 코드(이 코드 때문에 리덕션을 위해서는 threadsperblock은 2의 제곱수이어야 한다)
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
			//__syncthreads(); //이렇게 넣게 되면 오류가 발생 ->120p 에 자세한 내용이 나와있음
		__syncthreads(); //cache에 모든 스레드가 값을 기록하고 나서 다음 반복을 시작
		i /= 2;
	}

	if (cacheIndex == 0)
	{
		c[blockIdx.x] = cache[0];
	}
}

void test1()
{

	//double d = 12.345345; printf("%.4g\n\n\n", d); //%.6g -> 숫자가 6개인 실수로 나타내줌(12.35) (만약에 %.4g 면 숫자가 4개인 실수로 나타내줌 (12.345))

	float* a, * b, c = 0, * partial_c;
	float* dev_a, * dev_b, * dev_c, * dev_partial_c;

	//a = new float[N]; // c++ 에서 malloc과 같은것 
	a = (float*)malloc(sizeof(float) * N); //와 같음
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
		printf("%g ", partial_c[i]);      //  .숫자g 와 g 의 차이를 알아보기 위함
		printf(" %.6g\n", partial_c[i]);
	}
	#define sum_squares(x) (x*(x+1)*(2*x+1)/6) // 1부터 x까지 각 제곱의 합
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
//에서는 3번째 줄에 도달했을 때 짝수 번의 스레드들은 if 구문을 만족하지 않기 때문에 
//오직 홀수 번의 스레드들만이 해당 라인을 실행할 것임.
//이를 수행하는 동안 짝수번의 스레드들은 아무것도 하지 않음. but 모두 연산을 동작함 -> 그래서 느려짐


//다시 정리를 해보면 GPU에서 분기 ( if문 )이 나쁜 ( 느린 ) 이유는 조건문을 연산하는 것 자체가 느린 것이 아니라
//GPU의 Warp단위 쓰레드들이 분기점에서 두 개의 분기 모두를 연산해야하기 때문이다.


