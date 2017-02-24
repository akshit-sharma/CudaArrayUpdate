#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<time.h>
#include <stdlib.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define numThreads 1000
#define arraySize 3000000
#define blockSize 65535


void printArray(int* array, long int size)
{
	for (long int i = 0; i < size; i++)
		if((i+1) % 100000 == 0)
		printf("%d ", *(array + i));
}

__global__ void incrementNaive(int *g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<arraySize)
		g[i] = 2;
}

int main()
{
	clock_t start, end;
	printf("%d total threads in %d blocks writing into %d array elements\n", numThreads, numThreads / blockSize, arraySize);
	int * h_array;
	int *d_array;

	h_array = (int *)malloc(sizeof(int) * arraySize);
	const size_t arrayBytes = arraySize * sizeof(int);
	
	printf("Max int is %d and size of one is %d \n", INT_MAX, sizeof(int));
	printf("Using size : %ld B or %ld KB or %ld MB or %ld GB \n", arrayBytes,
		arrayBytes >> 10,
		arrayBytes >> 20,
		arrayBytes >> 30
	);

	for(size_t i=0; i<arraySize; i++)
	{
		h_array[i] = i + 1;
	}

	cudaMalloc((void**)&d_array, arrayBytes);
	cudaMemcpy(d_array, h_array, arrayBytes, cudaMemcpyHostToDevice);
	
	start = clock();
	incrementNaive<<<blockSize, numThreads >>> (d_array);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	gpuErrchk(cudaDeviceSynchronize());
	end = clock();
	
	gpuErrchk(cudaMemcpy(h_array, d_array, arrayBytes, cudaMemcpyDeviceToHost));
	
	printArray(h_array, arraySize);
	
	printf("\nTime elapsed=%7.4lf sec\n", (double)(end - start) / CLOCKS_PER_SEC);
	cudaFree(d_array);
	
	return 0;
}
