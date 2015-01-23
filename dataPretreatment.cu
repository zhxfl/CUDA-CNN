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

	cuMatrix<double>* aver = new cuMatrix<double>(n_rows, n_cols, n_channels);

	for (int imgId = 0; imgId < trainX.size(); imgId++) {
		int len = trainX[0]->getLen();
		for (int i = 0; i < len; i++) {
			aver->hostData[i] += trainX[imgId]->hostData[i];
		}
	}

	for(int i = 0; i < aver->getLen(); i++){
		int len = trainX.size();
		aver->hostData[i] /= len;
	}

	for (int imgId = 0; imgId < trainX.size(); imgId++) {
		int len = trainX[0]->getLen();
		for (int i = 0; i < len; i++) {
			 trainX[imgId]->hostData[i] -= aver->hostData[i];
		}
	}

	aver->cpuClear();


	for (int imgId = 0; imgId < testX.size(); imgId++) {
		int len = testX[0]->getLen();
		for (int i = 0; i < len; i++) {
			aver->hostData[i] += testX[imgId]->hostData[i];
		}
	}

	for(int i = 0; i < aver->getLen(); i++){
		int len = testX.size();
		aver->hostData[i] /= len;
	}

	for (int imgId = 0; imgId < testX.size(); imgId++) {
		int len = testX[0]->getLen();
		for (int i = 0; i < len; i++) {
			testX[imgId]->hostData[i] -= aver->hostData[i];
		}
	}

	delete aver;
}
