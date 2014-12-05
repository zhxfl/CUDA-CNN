#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include "net.cuh"
#include "cuMatrix.h"
#include "util.h"
#include "readMnistData.h"
#include "cuDistortion.cuh"
#include "Config.h"
#include "cuMatrixVector.h"

/*init cublas Handle*/
bool init(cublasHandle_t& handle)
{
	cublasStatus_t stat;
	stat = cublasCreate(&handle);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf ("init: CUBLAS initialization failed\n");
		exit(0);
	}
	return true;
}

void runMnist(){
	const int nclasses = 10;

 	/*state and cublas handle*/
 	cublasHandle_t handle;
	init(handle);
	
 	/*read the data from disk*/
	cuMatrixVector<double>trainX;
	cuMatrixVector<double>testX;
 	cuMatrix<double>* trainY, *testY;

	Config::instance();
 	readMnistData(trainX, trainY, "mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", 60000, 1);
 	readMnistData(testX , testY,  "mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte",   10000, 1);
 
 	/*build CNN net*/
 	int ImgSize = trainX[0]->rows;
 	int nsamples = trainX.size();
 	std::vector<cuCvl> ConvLayers;
 	std::vector<cuFll> HiddenLayers;
 	cuSMR smr;
 	int batch = 200;
	double start,end;
	int cmd;
	cuInitDistortionMemery(batch, ImgSize);
	printf("random init input 0\nRead from file input 1\n");

	scanf("%d", &cmd);
 	if(cmd == 0)
		cuConvNetInitPrarms(ConvLayers, HiddenLayers, smr, ImgSize, nsamples, nclasses);
	else if(cmd == 1)
		cuReadConvNet(ConvLayers, HiddenLayers, smr, ImgSize, nsamples, "net.txt", nclasses);

	cuInitCNNMemory(batch, trainX, testX, ConvLayers,HiddenLayers, smr, ImgSize, nclasses);
	start = clock();
	cuTrainNetwork(trainX, trainY, ConvLayers, HiddenLayers, smr, 4e-4, testX, testY, nsamples, batch, ImgSize, nclasses, handle);
	end = clock();
	printf("trainning time %lf\n", (end - start) / CLOCKS_PER_SEC);
}


void cuPredict()
{
	const int nclasses = 10;
	const int ImgSize = 28;

	/*state and cublas handle*/
	cublasHandle_t handle;
	init(handle);

	/*read the data from disk*/
	cuMatrixVector<double> trainX;
	cuMatrixVector<double> testX;
	cuMatrix<double>* trainY, *testY;

	int num1 = readMnistData(trainX, trainY, "mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", 60000, 0);
	int num2 = readMnistData(testX , testY,  "mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte",   10000, 1);

	printf("train DataSize = %d, test DataSize = %d\n", num1, num2);

	/*build CNN net*/
	int imgDim = trainX[0]->rows;
	int nsamples = trainX.size();
	std::vector<cuCvl> ConvLayers;
	std::vector<cuFll> HiddenLayers;
	cuSMR smr;
	int batch = 200;

	cuInitDistortionMemery(batch, ImgSize);

	//char* path[] = {"4_9962", "3_9951"};
	//char* initPath[] = {"P4.txt",  "P3.txt"};
	//char* path[] = {"p/p1/net.txt", "p/p2/net.txt", "p/p3/net.txt", "p/p4/net.txt", "p/p5/net.txt", "p/p6/net.txt", "p/p7/net.txt"};
	//char* initPath[] = {"p/p1/Pnet.txt", "p/p2/Pnet.txt", "p/p3/Pnet.txt", "p/p4/Pnet.txt", "p/p5/Pnet.txt", "p/p6/Pnet.txt", "p/p7/Pnet.txt"};

	// 	char* path[] = {"p10_20_256_256/p1/net.txt", "p10_20_256_256/p2/net.txt", "p10_20_256_256/20w/net.txt", "p10_20_256_256/dl/net.txt", "p10_20_256_256/jk/net.txt",
	// 		"p10_20_256_256/lw/net.txt", "p10_20_256_256/lyx/net.txt", "p10_20_256_256/tdx/net.txt", "p10_20_256_256/wo/net.txt",
	// 		"p10_20_256_256/xh/net.txt", "p10_20_256_256/yy/net.txt","p10_20_256_256/wy/net.txt","p10_20_256_256/hk/net.txt"};
	// 	char* initPath[] = {"p10_20_256_256/p1/Pnet.txt", "p10_20_256_256/p2/Pnet.txt", "p10_20_256_256/20w/Pnet.txt", "p10_20_256_256/dl/Pnet.txt", "p10_20_256_256/jk/Pnet.txt",
	// 		"p10_20_256_256/lw/Pnet.txt", "p10_20_256_256/lyx/Pnet.txt", "p10_20_256_256/tdx/Pnet.txt", "p10_20_256_256/wo/Pnet.txt",
	// 		"p10_20_256_256/xh/Pnet.txt", "p10_20_256_256/yy/Pnet.txt", "p10_20_256_256/wy/Pnet.txt", "p10_20_256_256/hk/Pnet.txt"};

 	char* path[] = {"p10_20_256_256/hk/2/net.txt",
 		"p10_20_256_256/p1/net.txt", "p10_20_256_256/20w/net.txt", "p10_20_256_256/dl/net.txt",
 		"p10_20_256_256/lw/net.txt", "p10_20_256_256/lw/1/net.txt", "p10_20_256_256/tdx/net.txt",/*"p10_20_256_256/lyx/net.txt",*/  /*"p10_20_256_256/yy/net.txt",*/
 		"p10_20_256_256/yy/1/net.txt", /*"p10_20_256_256/xh/1/net.txt",*/
 		"p10_20_256_256/dl/1/net.txt", "p10_20_256_256/hk/1/net.txt",
 		"p10_20_256_256/xh/net.txt",
 		"p10_20_256_256/20w/2/net.txt",
		"1/jk/net.txt",  "1/dl/net.txt", "1/lw/net.txt", "1/xh/net.txt"};
 	char* initPath[] = {"p10_20_256_256/hk/2/Pnet.txt",
 		"p10_20_256_256/p1/Pnet.txt", "p10_20_256_256/20w/Pnet.txt", "p10_20_256_256/dl/Pnet.txt",
 		"p10_20_256_256/lw/Pnet.txt", "p10_20_256_256/lw/1/Pnet.txt", "p10_20_256_256/tdx/Pnet.txt", /*"p10_20_256_256/lyx/Pnet.txt",*/  /*"p10_20_256_256/yy/Pnet.txt",*/
 		"p10_20_256_256/yy/1/Pnet.txt",/*"p10_20_256_256/xh/1/Pnet.txt",*/
 		"p10_20_256_256/dl/1/Pnet.txt","p10_20_256_256/hk/1/Pnet.txt",
 		"p10_20_256_256/xh/Pnet.txt",
 		"p10_20_256_256/20w/2/Pnet.txt",
		"1/jk/Pnet.txt", "1/dl/Pnet.txt","1/lw/Pnet.txt","1/xh/Pnet.txt"};

//  	char* path[]    = {"1/jk/net.txt",  "1/dl/net.txt", "1/lw/net.txt", "1/xh/net.txt", "p10_20_256_256/lw/1/net.txt"};
//  	char*initPath[] = {"1/jk/Pnet.txt", "1/dl/Pnet.txt","1/lw/Pnet.txt","1/xh/Pnet.txt","p10_20_256_256/lw/1/Pnet.txt"};
	int numPath = sizeof(path) / sizeof(char*);
	int * mark  = new int[1 << numPath];
	memset(mark, 0, sizeof(int) * (1 << numPath));
	std::vector<int>vCorrect;
	std::vector<cuMatrix<double>*>vPredict;

	for(int i = 0; i < numPath; i++)
	{
		vPredict.push_back(new cuMatrix<double>(trainY->getLen(), 1));
		cuReadConvNet(ConvLayers, HiddenLayers, smr, imgDim, nsamples, path[i], nclasses);
		cuInitCNNMemory(batch, trainX, testX, ConvLayers, HiddenLayers, smr, ImgSize, nclasses);
		int cur = cuPredictNetwork(trainX, trainY, ConvLayers, HiddenLayers, smr, 3e-3, testX, testY, vPredict[i],ImgSize, nsamples, batch, ImgSize, nclasses, handle);
		cuFreeCNNMemory(batch, trainX, testX, ConvLayers,HiddenLayers, smr);
		cuFreeConvNet(ConvLayers, HiddenLayers, smr);
		vCorrect.push_back(cur);
		printf("%d %s\n", cur, path[i]);
	}
	int max = -1;
	int val = 1;
	int cur;
	return;
 	for(int m = (1 << numPath) - 1; m >= 1; m--)
 	{
		//m = 14653;
 		//m = 205;
 		if(mark[m] != 0)
 		{
 			cur = mark[m];
 			for(int i = 0; i < numPath; i++)
 			{
 				if(!(m & (1 << i)))continue;
 				printf("%d %s\n", vCorrect[i], path[i]);
 			}
 		}
 		else
 		{
 			int v = 0;
 			for(int i = numPath - 1; i >= 0; i--)
 			{
 				if(!(m & (1 << i)))continue;
 				v = v | (1 << i);
 				cur = cuPredictAdd(vPredict[i], testY, batch, ImgSize, nclasses);
 				mark[v] = cur;
 				printf("%d %d %s\n", vCorrect[i], cur, path[i]);
 			}
 		}
 		if(cur >= max)
 		{
 			max = cur;
 			val = m;
 		}
 
 		cuClearCorrectCount();
 		printf("m = %d val = %d max = %d \n\n",m, val, max);
 	}

	cuClearCorrectCount();
	int m = val;
	for(int i = numPath - 1; i >= 0; i--)
	{
		if(!(m & (1 << i)))continue;
		cur = cuPredictAdd(vPredict[i], testY, batch, ImgSize, nclasses);
		printf("%d %d %s\n", vCorrect[i], cur, path[i]);
	}
	cuShowInCorrect(testX, testY, ImgSize, nclasses);
}

int main (void)
{
	runMnist();
	//cuPredict();
	return EXIT_SUCCESS;
}
