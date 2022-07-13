
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE (100 * 1024 * 1024)

int main()
{
	unsigned char* buffer = (unsigned char*)big_random_block(SIZE); // 100MB�� �������� �����͸� ����
	
	
	unsigned int histo[256];// ������ 8��Ʈ�� 256�� �������� � �ϳ��� ��(0x00~0xff)�� �� �� �����Ƿ� 
	//������׷��� �����Ϳ��� �� ���� ������ Ƚ���� ����ϱ� ���� 256���� ����Ұ� �ʿ�.

	//�� ����Ҹ� 0���� �ʱ�ȭ
	for (int i = 0; i < 256; i++)
	{
		histo[i] = 0;
	}

	//histo�� buffer[] �����Ϳ��� �����ϴ� �� ���� �󵵸� ����
	for (int i = 0; i < SIZE; i++)
	{
		histo[buffer[i]]++;
	}

	//��� ����ҿ� ���� ������׷� �հ谡 �츮�� ������ ������ Ȯ�����ش�.
	long histoCount = 0;
	for (int i = 0; i < 256; i++)
	{
		histoCount += histo[i];
	}
	printf("Histogram Sum: %ld\n", histoCount);

	free(buffer);
	return 0;
}