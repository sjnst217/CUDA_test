
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE (100 * 1024 * 1024)

int main()
{
	unsigned char* buffer = (unsigned char*)big_random_block(SIZE); // 100MB의 무작위의 데이터를 생성
	
	
	unsigned int histo[256];// 각각의 8비트는 256중 무작위로 어떤 하나의 값(0x00~0xff)이 될 수 있으므로 
	//히스토그램은 데이터에서 각 값이 등장한 횟수를 기록하기 위해 256개의 저장소가 필요.

	//각 저장소를 0으로 초기화
	for (int i = 0; i < 256; i++)
	{
		histo[i] = 0;
	}

	//histo에 buffer[] 데이터에서 등장하는 각 값의 빈도를 저장
	for (int i = 0; i < SIZE; i++)
	{
		histo[buffer[i]]++;
	}

	//모든 저장소에 대한 히스토그램 합계가 우리가 예상한 값인지 확인해준다.
	long histoCount = 0;
	for (int i = 0; i < 256; i++)
	{
		histoCount += histo[i];
	}
	printf("Histogram Sum: %ld\n", histoCount);

	free(buffer);
	return 0;
}