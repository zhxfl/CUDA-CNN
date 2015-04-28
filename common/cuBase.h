#ifndef __CU_BASE_CU_H__
#define __CU_BASE_CU_H__

#include <helper_functions.h>
#include <helper_cuda.h>
#include "util.h"

__device__ double atomicAdd(double* address, double val);
__device__ float d_nonLinearity(float val, int NONLIN);
__device__ float d_dnonLinearity(float val,int NONLIN);
__device__ void   swap(float& val1, float& val2);


__global__ void   g_dnonLinearity(float* delta, float*acti, int len, int NONLIN);
__global__ void   g_nonLinearity(float* inputs, int len, int NONLIN);

__global__ void g_vecAdd(float**_v_w, float** _wgrad,float** _w,
	float** _v_b, float** _bgrad, float** _b, 
	int lenw, int lenb,
	float momentum, float lratew, float lrateb);

__global__ void g_vecAdd(float*v_w, float*wgrad,float* w,
	float* v_b, float* bgrad, float* b, 
	int lenw, int lenb,
	float momentum, float lratew, float lrateb);

__global__ void g_getBgrad(float* softMaxDelta, float* bgrad, float* dropb, int batch);
__global__ void g_getBgrad(float* softMaxDelta, float* bgrad, int batch);

__global__ void g_getCost_3(float* cost,
	float** weight,
	float lambda, int wlen);

__global__ void g_getCost_2(float* cost,
	float* weight,
	float lambda, int len);

__global__ void g_getCost_1(float* softMaxP,
	float* groundTruth, float* cost, int*y, int rows, int cols, int batch);

/*
* function: cuMatrix(batch, size, channel) to cuMatrix(batch, size * channel, 1)
* blocks  : dim3(batch)
* threads : dim3(min(512, cuPool[poolidx]->cols))
*/
__global__ void g_convert(float* cuPool, float*cuPoolToFlActi, int batch, int size, int channel);

/*
function: g_preDeltaFormat
threads : <<<dim3(batch), dim3(512)>>> 
*/
__global__ void g_preDeltaFormat(float* cuPoolFlDelta, 
	float* cuPoolDelta, int batch, int size, int channels);

#endif


