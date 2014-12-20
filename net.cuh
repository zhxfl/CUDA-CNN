#ifndef _NET_CUH_
#define _NET_CUH_

#include "cuMatrix.h"
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cuMatrixVector.h"

/*卷积核*/
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

/*卷积层*/
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

/*全链接层*/
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

/*输出层，采用softmax回归*/
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
	function ：init network
	parameter：
	vector<Cvl> &ConvLayers      convolution
	vector<Ntw> &HiddenLayers	 hidden
	SMR &smr					 softmax
	int imgDim					 total pixel 
	int nsamples			     number of samples
*/
void cuConvNetInitPrarms(std::vector<cuCvl> &ConvLayers,
	std::vector<cuFll> &HiddenLayers,
	cuSMR &smr,
	int imgDim,
	int nsamples,
	int nclasses);

void cuReadConvNet(std::vector<cuCvl> &ConvLayers,
	std::vector<cuFll> &HiddenLayers, 
	cuSMR &smr, 
	int imgDim, 
	int nsamples, 
	char* path,
	int nclasses);

/*
	函数功能：神经网络的训练
	函数参数：
	vector<Mat> &x            ：输入的训练样本
	Mat &y                    ：训练样本对应的标签
	vector<Cvl> &CLayers      ：卷积层
	vector<Ntw> &HiddenLayers ：隐藏层
	SMR &smr				  ：softMax层
	double lambda			  ：？？？？
	vector<Mat>&testX		  ：测试集合
	Mat& testY				  ：测试集合标签
	int imgDim				  ：每个图片像素总和
	int nsamples              ：训练样本总数
*/

void cuTrainNetwork(cuMatrixVector<double>&x, 
	cuMatrix<int>*y , 
	std::vector<cuCvl> &CLayers,
	std::vector<cuFll> &HiddenLayers, 
	cuSMR &smr,
	double lambda, 
	cuMatrixVector<double>& testX,
	cuMatrix<int>* testY,
	int nsamples,
	int batch,
	int ImgSize,
	int nclasses,
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

void cuClearCorrectCount();

void cuFreeCNNMemory(
	int batch,
	cuMatrixVector<double>&trainX, 
	cuMatrixVector<double>&testX,
	std::vector<cuCvl>&ConvLayers,
	std::vector<cuFll>&HiddenLayers, 
	cuSMR &smr);

int cuPredictAdd(cuMatrix<double>* predict, cuMatrix<double>* testY, int batch, int ImgSize, int nclasses);

void cuShowInCorrect(cuMatrixVector<double>&testX, cuMatrix<double>* testY, int ImgSize, int nclasses);

#endif