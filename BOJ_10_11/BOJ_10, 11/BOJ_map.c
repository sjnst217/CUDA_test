
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int static compare(const void* first, const void* second)
{
	if (*(int*)first > *(int*)second)
	{
		return 1;
	}
	else if (*(int*)first < *(int*)second)
	{
		return -1;
	}
	else
	{
		return 0;
	}
}

int binarysearch(int data[], int n, int key) {
	int low, high;
	int mid;

	low = 0;
	high = n - 1;
	while (low <= high) {
		mid = (low + high) / 2;
		if (key == data[mid])
		{
			return mid;
		}
		else if (key < data[mid])
		{
			high = mid - 1;
		}
		else if (key > data[mid])
		{
			low = mid + 1;
		}
	}
	return -1;
}

//////
void test_10815()
{
	int N = 0;
	int* num_N;
	int* num_M;
	int M;
	int i = 0;
	int ch = 0;
	int idx;

	scanf("%d", &N);

	num_N = (int*)malloc(N * sizeof(int));

	for (i = 0; i < N; i++)
	{
		scanf("%d", &num_N[i]);
	}

	scanf("%d", &M);

	num_M = (int*)malloc(M * sizeof(int));

	for (i = 0; i < M; i++)
	{
		scanf("%d", &num_M[i]);
	}

	qsort(num_N, N, sizeof(int), compare);

	for (i = 0; i < M; i++)
	{
		idx = 0;

		idx = binarysearch(num_N, N, num_M[i]);
		if (idx != -1)
		{
			printf("1 ");
		}
		else
		{
			printf("0 ");
		}
	}
}

int main()
{
	test_10815();

	getchar();
	return 0;
}