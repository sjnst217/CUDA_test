
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


typedef struct _AGE_NAME_ {
	int age;
	char name[101];
}AGE_NAME;

void test_10814()
{
	int num = 0;

	scanf("%d", &num);

	AGE_NAME* arr = (AGE_NAME*)malloc(num * sizeof(AGE_NAME));
	int temp = 0;
	char* str_temp = 0;

	for (int i = 0; i < num; i++)
	{
		scanf("%d %s", &arr[i].age, &arr[i].name);
	}

	qsort(arr, num, sizeof(arr[0]), compare);

	for (int i = 0; i < num; i++)
	{
		printf("%d %s\n", arr[i].age, arr[i].name);
	}
}

void test_18870()
{
	int num;
	int ch1 = 0;
	int ch2 = 0;
	int temp = 0;

	scanf("%d", &num);

	int* X = (int*)malloc(num * sizeof(int));
	int* cpy_X = (int*)calloc(num, sizeof(int));
	int* us_X = (int*)calloc(num, sizeof(int));

	for (int i = 0; i < num; i++)
	{
		scanf("%d", &X[i]);
	}

	for (int i = 0; i < num; i++)
	{
		cpy_X[i] = X[i];
	}

	qsort(cpy_X, num, sizeof(cpy_X[0]), compare);


	int k = 0;
	for (int i = 0; i < num; i++)
	{
		if (i == num - 1)
		{
			us_X[k] = cpy_X[i];
			k++;
			break;
		}
		if (cpy_X[i] < cpy_X[i + 1])
		{
			us_X[k] = cpy_X[i];
			k++;
		}
	}

	for (int i = 0; i < num; i++)
	{
		temp = binarysearch(us_X, k, X[i]);
		printf("%d ", temp);
	}
}

int main()
{
	test_18870();

	getchar();
	return 0;
}