#ifndef __MEMORY_MONITOR_H__
#define __MEMORY_MONITOR_H__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "helper_cuda.h"
#include <cuda_runtime.h>
#include <map>
class MemoryMonitor
{
public:
	static MemoryMonitor* instance(){
		static MemoryMonitor* monitor = new MemoryMonitor();
		return monitor;
	}
	void* cpuMalloc(int size);
	cudaError_t gpuMalloc(void** devPtr, int size);
	MemoryMonitor(): gpuMemory(0), cpuMemory(0){}
	void printCpuMemory(){printf("total malloc cpu memory %lfMb\n", cpuMemory / 1024 / 1024);}
	void printGpuMemory(){printf("total malloc gpu memory %lfMb\n", gpuMemory / 1024 / 1024);}
	void freeGpuMemory(void* ptr);
	void freeCpuMemory(void* ptr);
private:
	double cpuMemory;
	double gpuMemory;
	std::map<void*, double>cpuPoint;
	std::map<void*, double>gpuPoint;
};
#endif
