#include "dataPretreatment.cuh"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuMatrix.h"
#include "cuMatrixVector.h"


__global__ void g_getAver(double ** input1,
	double** input2, 
	double* aver,
	int num_of_input1, 
	int num_of_input2,
	int imgSize)
{
	for(int j = 0; j < num_of_input1; j++)
	{
		for(int i = 0; i < imgSize; i += blockDim.x)
		{
			int idx = threadIdx.x + i;
			if(idx < imgSize)
			{
				aver[idx] += input1[j][idx];
			}
		}
	}

	__syncthreads();

	for(int j = 0; j < num_of_input2; j++)
	{
		for(int i = 0; i < imgSize; i += blockDim.x)
		{
			int idx = threadIdx.x + i;
			if(idx < imgSize)
			{
				aver[idx] += input2[j][idx];
			}
		}
	}

	__syncthreads();

	for(int i = 0; i < imgSize; i += blockDim.x)
	{
		int idx = i + threadIdx.x;
		if(idx < imgSize)
		{
			aver[idx] /= (num_of_input1 + num_of_input2);
		}
	}

	__syncthreads();


	for(int j = 0; j < num_of_input1; j++)
	{
		for(int i = 0; i < imgSize; i += blockDim.x)
		{
			int idx = threadIdx.x + i;
			if(idx < imgSize)
			{
				input1[j][idx] -= aver[idx];
			}
		}
	}

	__syncthreads();

	for(int j = 0; j < num_of_input2; j++)
	{
		for(int i = 0; i < imgSize; i += blockDim.x)
		{
			int idx = threadIdx.x + i;
			if(idx < imgSize)
			{
				input2[j][idx] -= aver[idx];
			}
		}
	}
}

void 
	preProcessing(cuMatrixVector<double>&trainX, cuMatrixVector<double>&testX)
{
	int n_rows     = trainX[0]->rows;
	int n_cols     = trainX[0]->cols;
	int n_channels = trainX[0]->channels;
	int num_of_images = trainX.size() + testX.size();

	cuMatrix<double>* aver = new cuMatrix<double>(n_rows, n_cols, n_channels);
	aver->gpuClear();

	g_getAver<<<dim3(1), dim3(512)>>>(trainX.m_devPoint,
		testX.m_devPoint,
		aver->devData,
		trainX.size(),
		testX.size(),
		trainX[0]->getLen());
	cudaDeviceSynchronize();
	delete aver;
}