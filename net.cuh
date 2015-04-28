#ifndef _NET_CUH_
#define _NET_CUH_

#include "common/cuMatrix.h"
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include "common/cuMatrixVector.h"

/*
 *function                : read the network weight from checkpoint
 * parameter              :
 * vector<Cvl> &ConvLayers: convolution
 * vector<Ntw> &FullLayers:	Full connect
 * SMR &smr			      :	softmax
 * int imgDim			  :	Image Size
 * int nsamples		      : number of samples
 */
void cuReadConvNet( 
	int imgDim, 
	char* path,
	int nclasses);

/*
 *function: trainning the network
*/

void cuTrainNetwork(cuMatrixVector<float>&x, 
	cuMatrix<int>*y ,
	cuMatrixVector<float>& testX,
	cuMatrix<int>* testY,
	int batch,
	int ImgSize,
	int nclasses,
	std::vector<float>&nlrate,
	std::vector<float>&nMomentum,
	std::vector<int>&epoCount,
	cublasHandle_t handle);

void buildNetWork(int trainLen, int testLen);

void cuFreeConvNet();

void cuFreeCNNMemory(
	int batch,
	cuMatrixVector<float>&trainX, 
	cuMatrixVector<float>&testX);

int cuVoteAdd(cuMatrix<int>*& voteSum, 
	cuMatrix<int>*& predict,
	cuMatrix<int>*& testY, 
	cuMatrix<int>*& correct,
	int nclasses);

#endif
