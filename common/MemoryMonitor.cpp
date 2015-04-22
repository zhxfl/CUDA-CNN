#include "MemoryMonitor.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "helper_cuda.h"

void* MemoryMonitor::cpuMalloc(int size){
	cpuMemory += size;
	void* p = malloc(size);
	cpuPoint[p] = size;
// 	if(size >= 1024 * 1024){
// 		printf("cpu malloc memory %lfMb\n", 1.0 * size / 1024 / 1024);
// 	}
	return p;
}

cudaError_t MemoryMonitor::gpuMalloc(void** devPtr, int size){
	gpuMemory += size;
	cudaError_t error = cudaMalloc(devPtr, size);
	gpuPoint[*devPtr] = size;
// 	if(size >= 1024 * 1024){
// 		printf("gpu malloc memory %lfMb\n", 1.0 * size / 1024 / 1024);
// 	}
	return error;
}

void MemoryMonitor::freeGpuMemory(void* ptr){
	if(gpuPoint.find(ptr) != gpuPoint.end()){
// 		if(gpuPoint[ptr] >= 1024 * 1024){
// 			//printf("free gpu memory %lfMb\n", gpuPoint[ptr] / 1024 / 1024);
// 		}
		gpuMemory -= gpuPoint[ptr];
		cudaFree(ptr);
		gpuPoint.erase(ptr);
	}
}


void MemoryMonitor::freeCpuMemory(void* ptr)
{
	if(cpuPoint.find(ptr) != cpuPoint.end()){
// 		if(cpuPoint[ptr] >= 1024 * 1024){
// 			printf("free cpu memory %lfMb\n", cpuPoint[ptr] / 1024 / 1024);
// 		}
		cpuMemory -= cpuPoint[ptr];
		free(ptr);
		cpuPoint.erase(ptr);
	}
}