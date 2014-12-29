#ifndef _NET_CUH_
#define _NET_CUH_

#include "cuMatrix.h"
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cuMatrixVector.h"


/*Convolution layer kernel*/
typedef struct cuConvKernel{
	cuMatrix<double>* W;
	cuMatrix<double>* b;
	cuMatrix<double>* Wgrad;
	cuMatrix<double>* bgrad;
	void clear()
	{
		delete W;
		delete b;
		delete Wgrad;
		delete bgrad;
	}
}cuConvK;

/*Convolution layer*/
typedef struct cuConvLayer{
	std::vector<cuConvK> layer;
	cuMatrixVector<double>* w;
	cuMatrixVector<double>* wgrad;
	cuMatrixVector<double>* b;
	cuMatrixVector<double>* bgrad;

	void init();
	void clear(){
		for(int i = 0; i < layer.size(); i++) 
			layer[i].clear();
		layer.clear();

		delete w;
		delete wgrad;
		delete b;
		delete bgrad;
	}
}cuCvl;

/*Full Connnect Layer*/
typedef struct cuFullLayer{
	cuMatrix<double>* W;
	cuMatrix<double>* b;
	cuMatrix<double>* Wgrad;
	cuMatrix<double>* bgrad;
	cuMatrix<double>* dropW;
	cuMatrix<double>* afterDropW;
	void clear()
	{
		delete W;
		delete b;
		delete Wgrad;
		delete bgrad;
		delete dropW;
		delete afterDropW;
	}
}cuFll;

/*SoftMax*/
typedef struct cuSoftmaxRegession{
	cuMatrix<double>* Weight;
	cuMatrix<double>* Wgrad;
	cuMatrix<double>* b;
	cuMatrix<double>* bgrad;
	cuMatrix<double>* cost;
	void clear()
	{
		delete Weight;
		delete Wgrad;
		delete b;
		delete bgrad;
		delete cost;
	}
}cuSMR;

/*
 * function               : init network
 * parameter              :
 * vector<Cvl> &ConvLayers: convolution
 * vector<Ntw> &FullLayers:	Full connect
 * SMR &smr			      :	softmax
 * int imgDim			  :	Image Size
*/
void cuConvNetInitPrarms(std::vector<cuCvl> &ConvLayers,
	std::vector<cuFll> &FullLayers,
	cuSMR &smr,
	int imgDim,
	int nclasses);

/*
 *function                : read the network weight from checkpoint
 * parameter              :
 * vector<Cvl> &ConvLayers: convolution
 * vector<Ntw> &FullLayers:	Full connect
 * SMR &smr			      :	softmax
 * int imgDim			  :	Image Size
 * int nsamples		      : number of samples
 */
void cuReadConvNet(std::vector<cuCvl> &ConvLayers,
	std::vector<cuFll> &FullLayers,
	cuSMR &smr, 
	int imgDim, 
	char* path,
	int nclasses);

/*
 *function: trainning the network
*/

void cuTrainNetwork(cuMatrixVector<double>&x, 
	cuMatrix<int>*y , 
	std::vector<cuCvl> &CLayers,
	std::vector<cuFll> &FULLLayers,
	cuSMR &smr,
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
	std::vector<cuCvl>& ConvLayers,
	std::vector<cuFll>& HiddenLayers,
	cuSMR& smr,
	int ImgSize,
	int nclasses);

void cuFreeConvNet(std::vector<cuCvl> &ConvLayers,
	std::vector<cuFll> &HiddenLayers,
	cuSMR &smr);

void cuFreeCNNMemory(
	int batch,
	cuMatrixVector<double>&trainX, 
	cuMatrixVector<double>&testX,
	std::vector<cuCvl>&ConvLayers,
	std::vector<cuFll>&HiddenLayers, 
	cuSMR &smr);

#endif
