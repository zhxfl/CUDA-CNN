#ifndef __CU_BASE_CU_H__
#define __CU_BASE_CU_H__

#include <helper_functions.h>
#include <helper_cuda.h>
#include "util.h"

__device__ double atomicAdd(double* address, double val);

__device__ double d_nonLinearity(double val, int NONLIN);
__device__ double d_dnonLinearity(double val,int NONLIN);

__global__ void   g_dnonLinearity(double* delta, double*acti, int len, int NONLIN);
__global__ void   g_nonLinearity(double* inputs, int len, int NONLIN);
__device__ void   swap(double& val1, double& val2);

__global__ void g_vecAdd(double**_v_w, double** _wgrad,double** _w,
	double** _v_b, double** _bgrad, double** _b, 
	int lenw, int lenb,
	double momentum, double lrate);

__global__ void g_vecAdd(double*v_w, double*wgrad,double* w,
	double* v_b, double* bgrad, double* b, 
	int lenw, int lenb,
	double momentum, double lrate);

__global__ void g_getBgrad(double* softMaxDelta, double* bgrad, double* dropb, int batch);
__global__ void g_getBgrad(double* softMaxDelta, double* bgrad, int batch);

__global__ void g_getCost_3(double* cost,
	double** weight,
	double lambda, int wlen);

__global__ void g_getCost_2(double* cost,
	double* weight,
	double lambda, int len);

__global__ void g_getCost_1(double* softMaxP,
	double* groundTruth, double* cost, int*y, int rows, int cols, int batch);

/*
* function: cuMatrix(batch, size, channel) to cuMatrix(batch, size * channel, 1)
* blocks  : dim3(batch)
* threads : dim3(min(512, cuPool[poolidx]->cols))
*/
__global__ void g_convert(double* cuPool, double*cuPoolToFlActi, int batch, int size, int channel);

/*
function: g_preDeltaFormat
threads : <<<dim3(batch), dim3(512)>>> 
*/
__global__ void g_preDeltaFormat(double* cuPoolFlDelta, 
	double* cuPoolDelta, int batch, int size, int channels);

#endif


