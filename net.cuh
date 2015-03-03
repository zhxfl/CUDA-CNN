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

void cuTrainNetwork(cuMatrixVector<double>&x, 
	cuMatrix<int>*y ,
	cuMatrixVector<double>& testX,
	cuMatrix<int>* testY,
	int batch,
	int ImgSize,
	int nclasses,
	std::vector<double>&nlrate,
	std::vector<double>&nMomentum,
	std::vector<int>&epoCount,
	cublasHandle_t handle);

void cuInitCNNMemory(
	int batch,
	cuMatrixVector<double>& trainX, 
	cuMatrixVector<double>& testX,
	int ImgSize,
	int nclasses);

void cuFreeConvNet();

void cuFreeCNNMemory(
	int batch,
	cuMatrixVector<double>&trainX, 
	cuMatrixVector<double>&testX);


/*
*	get one network's vote result
*/

int voteTestDate(
	cuMatrixVector<double>&testX,
	cuMatrix<int>* testY,
	cuMatrix<int>*& vote,
	int batch,
	int ImgSize,
	int nclasses,
	cublasHandle_t handle);

int cuVoteAdd(cuMatrix<int>*& voteSum, 
	cuMatrix<int>*& predict,
	cuMatrix<int>*& testY, 
	cuMatrix<int>*& correct,
	int nclasses);

#endif
