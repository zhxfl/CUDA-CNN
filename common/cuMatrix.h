#ifndef _CU_MATRIX_H_
#define _CU_MATRIX_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "helper_cuda.h"
#include "MemoryMonitor.h"

/*rows-major*/
template <class T>
class cuMatrix
{
public:
	/*constructed function with hostData*/
	cuMatrix(T *_data, int _n,int _m, int _c):rows(_n), cols(_m), channels(_c){
		cudaError_t cudaStat;
		/*malloc host data*/
		hostData = (T*)MemoryMonitor::instance()->cpuMalloc(cols * rows * channels * sizeof(*hostData));
		if(!hostData) {
			printf("cuMatrix:cuMatrix host memory allocation failed\n");
			exit(0);
		}

		/*deep copy */
		memcpy(hostData, _data, sizeof(*hostData) * cols * rows * channels);

		/*malloc device data*/
		cudaStat = MemoryMonitor::instance()->gpuMalloc((void**)&devData, cols * rows * channels * sizeof(*devData));
		if(cudaStat != cudaSuccess) {
			printf ("cuMatrix::cuMatrix device memory allocation failed\n");
			exit(0);
		}
	}
	/*constructed function with rows and cols*/
	cuMatrix(int _n,int _m, int _c):rows(_n), cols(_m), channels(_c){
		cudaError_t cudaStat;

		/*malloc host data*/
		hostData = (T*)MemoryMonitor::instance()->cpuMalloc(cols * rows * channels * sizeof(*hostData));
		if(!hostData) {
			printf("cuMatrix::cuMatrix host memory allocation failed\n");
			exit(0);
		}

		memset(hostData, 0, cols * rows * channels * sizeof(*hostData));

		//malloc device data
		cudaStat = MemoryMonitor::instance()->gpuMalloc((void**)&devData, cols * rows * channels * sizeof(*devData));
		if(cudaStat != cudaSuccess) {
			printf ("cuMatrix::cuMatrix device memory allocation failed\n");
			exit(0);
		}

		cudaStat = cudaMemset(devData, 0, sizeof(*devData) * cols * rows * channels);
		if(cudaStat != cudaSuccess) {
			printf ("cuMatrix::cuMatrix device memory cudaMemset failed\n");
			exit(0);
		}
	}

	/*free cuda memery*/
	void freeCudaMem(){
		MemoryMonitor::instance()->freeGpuMemory(devData);
		devData = 0;
	}

	/*destruction function*/
	~cuMatrix(){
		MemoryMonitor::instance()->freeCpuMemory(hostData);
		MemoryMonitor::instance()->freeGpuMemory(devData);
	}

	/*copy the device data to host data*/ 
	void toCpu(){
		cudaError_t cudaStat;

		cudaStat = cudaMemcpy (hostData, devData, sizeof(*devData) * cols * rows * channels, cudaMemcpyDeviceToHost);

		if(cudaStat != cudaSuccess) {
			printf("cuMatrix::toCPU data download failed\n");
			MemoryMonitor::instance()->freeGpuMemory(devData);
			exit(0);
		} 
	}

	/*copy the host data to device data*/
	void toGpu(){
		cudaError_t cudaStat;

		cudaStat = cudaMemcpy (devData, hostData, sizeof(*devData) * cols * rows * channels, cudaMemcpyHostToDevice);

		if(cudaStat != cudaSuccess) {
			printf ("cuMatrix::toGPU data upload failed\n");
			MemoryMonitor::instance()->freeGpuMemory(devData);
			exit(0);
		}
	}

	/*copy the host data to device data with cuda-streams*/
	void toGpu(cudaStream_t stream1){
		checkCudaErrors(cudaMemcpyAsync(devData, hostData, sizeof(*devData) * cols * rows * channels, cudaMemcpyHostToDevice, stream1));
	}
	
	/*set all device memory to be zeros*/
	void gpuClear(){
		cudaError_t cudaStat;
		cudaStat = cudaMemset(devData,0,sizeof(*devData) * cols * rows * channels);
		if(cudaStat != cudaSuccess) {
			printf ("device memory cudaMemset failed\n");
			exit(0);
		}
	}

	void cpuClear(){
		memset(hostData, 0, cols * rows * channels * sizeof(*hostData));
	}

	/*set  value*/
	void set(int i, int j, int k, T v){
		hostData[(i * cols + j) + cols * rows * k] = v;
	}

	/*get value*/
	T get(int i, int j, int k){
		return hostData[(i * cols + j) + cols * rows * k];
	}

	/*get the number of values*/
	int getLen(){
		return rows * cols * channels;
	}

	/*get rows * cols*/
	int getArea(){
		return rows * cols;
	}

	/*host data*/
	T *hostData;

	/*device data*/
	T *devData;

	/*column*/
	int cols;

	/*row*/
	int rows;

	/*channels*/
	int channels;
};


/*matrix multiply*/
/*z = x * y*/
void matrixMul   (const cuMatrix<double>* x, const cuMatrix<double>*y, cuMatrix<double>*z);
/*z = T(x) * y*/
void matrixMulTA (const cuMatrix<double>* x, const cuMatrix<double>*y, cuMatrix<double>*z);
/*z = x * T(y)*/
void matrixMulTB (const cuMatrix<double>* x, const cuMatrix<double>*y, cuMatrix<double>*z);
#endif
