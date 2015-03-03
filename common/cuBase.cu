#include "cuBase.h"

__device__ double d_nonLinearity(double val, int NONLIN){
	if(NONLIN == NL_RELU)
	{
		if(val < 0.0) return 0.0;
		else return val;
	}
	else if(NONLIN == NL_TANH)
	{
		return tanh(val * 2.0 / 3.0) * 1.7159;
	}
	else
	{
		return val;
	}
}

__device__ double d_dnonLinearity(double val,int NONLIN){
	if(NONLIN == NL_RELU)
	{
		if(val > 0.0) return 1.0;
		else return 0.0;
	}
	else if(NONLIN == NL_TANH)
	{
		double res = 1.7159;
		double temp = val * val / 1.7159;
		res = (res - temp) * 2.0 / 3.0;
		return res;
	}else 
	{
		return val;
	}
}

__global__ void g_dnonLinearity(double* delta, double*acti, int len, int NONLIN)
{
	for(int i = 0; i < len; i += gridDim.x * blockDim.x)
	{
		int id = blockDim.x * blockIdx.x + threadIdx.x + i;
		if(id < len)
		{	
			delta[id] *= d_dnonLinearity(acti[id], NONLIN);
		}
	}
}

__global__ void g_nonLinearity(double* inputs, int len, int NONLIN)
{
	for(int i = 0; i < len; i += gridDim.x * blockDim.x)
	{
		int id = blockDim.x * blockIdx.x + threadIdx.x + i;
		if(id < len)
		{	
			inputs[id] = d_nonLinearity(inputs[id], NONLIN);
		}
	}
}

__device__ double atomicAdd(double* address, double val)
{ 	
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
			__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__device__ void swap(double& val1, double& val2){
	double tmp = val1;
	val1 = val2;
	val2 = tmp;
}


__global__ void g_vecAdd(double**_v_w, double** _wgrad,double** _w,
	double** _v_b, double** _bgrad, double** _b, 
	int lenw, int lenb,
	double momentum, double lrate)
{
	double* v_w   = _v_w[blockIdx.x];
	double* wgrad = _wgrad[blockIdx.x];
	double* w     = _w[blockIdx.x];
	double* v_b   = _v_b[blockIdx.x];
	double* bgrad = _bgrad[blockIdx.x];
	double* b     = _b[blockIdx.x];

	int idx = threadIdx.x;
	for(int i = 0; i < lenw; i += blockDim.x)
	{
		int id = i + idx;
		if(id < lenw)
		{
			v_w[id] = v_w[id] * momentum + wgrad[id] * lrate;
			w[id] -= v_w[id];
		}
	}
	for(int i = 0; i < lenb; i += blockDim.x)
	{
		int id = i + idx;
		if(id < lenb)
		{
			v_b[id] = v_b[id] * momentum + bgrad[id] * lrate;
			b[id] -= v_b[id];
		}
	}
}

__global__ void g_vecAdd(double*v_w, double*wgrad,double* w,
	double* v_b, double* bgrad, double* b, 
	int lenw, int lenb,
	double momentum, double lrate)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i = 0; i < lenw; i += blockDim.x * gridDim.x)
	{
		int id = i + idx;
		if(id < lenw)
		{
			v_w[id] = v_w[id] * momentum + wgrad[id] * lrate;
			w[id] -= v_w[id];
		}
	}
	for(int i = 0; i < lenb; i += blockDim.x * gridDim.x)
	{
		int id = i + idx;
		if(id < lenb)
		{
			v_b[id] = v_b[id] * momentum + bgrad[id] * lrate;
			b[id] -= v_b[id];
		}
	}
}


__global__ void g_getCost_3(double* cost,
	double** weight,
	double lambda, int rows, int cols)
{
	extern __shared__ double _sum[];
	_sum[threadIdx.x] = 0;
	__syncthreads();
	double* w = weight[blockIdx.x];

	for(int i = 0; i < rows * cols; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < rows * cols)
		{
			_sum[threadIdx.x] += w[id] * w[id];
		}
	}

	int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < (len >> 1))
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = (len + 1) >> 1;
	}

	if(threadIdx.x == 0)
	{
		atomicAdd(cost, _sum[0] * lambda * 0.5);
	}
}


/*
*/
__global__ void g_getBgrad(double* softMaxDelta, double* bgrad, double* dropb, int batch)
{
	extern __shared__ double _sum[];
	_sum[threadIdx.x] = softMaxDelta[threadIdx.x * gridDim.x + blockIdx.x];

	int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < (len >> 1))
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = (len + 1) >> 1;
	}
	if(threadIdx.x == 0)
	{
		bgrad[blockIdx.x] = _sum[0] / batch;
		bgrad[blockIdx.x] *= dropb[blockIdx.x];
	}
}


/*
*/
__global__ void g_getBgrad(double* softMaxDelta, double* bgrad, int batch)
{
	extern __shared__ double _sum[];
	_sum[threadIdx.x] = softMaxDelta[threadIdx.x * gridDim.x + blockIdx.x];

	int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < (len >> 1))
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = (len + 1) >> 1;
	}
	if(threadIdx.x == 0)
	{
		bgrad[blockIdx.x] = _sum[0] / batch;
	}
}

/*
* function: getcost
*/
__global__ void g_getCost_1(double* softMaxP,
	double* groundTruth, double* cost, int*y, int rows, int cols, int batch)
{
	extern __shared__ double _sum[];
	int len = rows * cols;
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			groundTruth[id] = 0;
		}
	}
	__syncthreads();
	for(int i = 0; i < rows; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < rows)
		{
			int yy = y[id];
			groundTruth[id * cols + yy] = 1;
		}
	}
	_sum[threadIdx.x] = 0;
	__syncthreads();
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			_sum[threadIdx.x] += log(softMaxP[id]) * groundTruth[id];
		}
	}
	len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < (len >> 1))
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();
	if(threadIdx.x == 0)
	{
		cost[0] = -_sum[0] / batch;
	}
}


__global__ void g_getCost_2(double* cost,
	double* weight,
	double lambda, int len)
{
	extern __shared__ double _sum[];
	_sum[threadIdx.x] = 0;
	__syncthreads();
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			_sum[threadIdx.x] += weight[id] * weight[id];
		}
	}
	len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(threadIdx.x < (len >> 1))
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = (len + 1) >> 1;
	}
	if(threadIdx.x == 0)
	{
		cost[0] += _sum[0] * lambda * 0.5;
	}
}