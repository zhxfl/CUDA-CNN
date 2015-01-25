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

#endif


