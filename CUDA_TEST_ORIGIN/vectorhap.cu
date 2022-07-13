
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define arraySize  (33 * 1024) //->33792


void addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b) //GPU���� ����Ǵ� �ڵ�
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //������ �� ������ GPU ��(thread ���� ��ü ��)
    while(tid < arraySize) // arraySize ��ŭ�� ����� �ϱ� ���ؼ� �ݺ����� 
    {
        c[tid] = a[tid] + b[tid]; //�Ϲ����� ������ ����
        tid += blockDim.x * gridDim.x; //������ ������ arraySize���� ū tid�� �������� ���ϵ��� while���� �����Ű�� ����
    }
    // �Ʒ� �ڵ�� ����
    /*if (tid < arraySize)
    {
        c[tid] = a[tid] + b[tid];
    }*/
    // ���� ���� -> (GPU�� �ѹ��� ��� ������ 16384��ŭ�� tid������ ����ϰ� �� ������ 33729������ ��꿡��f c[tid]�� ������� �ʴ� �ڵ� �̹Ƿ�)
}

void test1()
{
    int a[arraySize] = { 0 };
    int b[arraySize] = { 0 };
    int c[arraySize] = { 0 };

    for (int i = 0; i < arraySize; i++)
    {
        a[i] = i;
        b[i] = i + i;
    }
    // Add vectors in parallel.
    addWithCuda(c, a, b, arraySize);
    printf("\n");
    
    bool success = true;
    for (int i = 0; i < arraySize; i++)
    {
        if ((a[i] + b[i]) != c[i])
        {
            printf("Error: %d + %d != %d\n", a[i], b[i], c[i]);
            success = false;
        }
    }
    if(success)
    {
        printf("We did it!\n");
    }
    
    for (int i = 16383; i < 16390; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    } //����� ������ �Ǿ����� Ȯ���ϴ� �ڵ�

   
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();
}

void test2()
{

}

int main()
{
    test1();

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;


    // Choose which GPU to run on, change this on a multi-GPU system.

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));

    

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    //       (block, thread)   (input or output) 
    addKernel <<<128, 128>>>(dev_c, dev_a, dev_b); //���⿡������ GPU����, �ѹ��� ó�� ������ ���� 16384

    // Check for any errors launching the kernel

    //cudaDeviceSynchronize();
    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);



    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

}
