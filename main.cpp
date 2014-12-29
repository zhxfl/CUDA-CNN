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
#include "cuTrasformation.cuh"
#include "Config.h"
#include "cuMatrixVector.h"
#include "readCIFAR10Data.h"
#include "Config.h"
#include "dataPretreatment.cuh"
#include "readChineseData.h"
#include "readCIFAR100Data.h"

void runMnist();
void runCifar10();
void runCifar100();
void runChinese();
bool init(cublasHandle_t& handle);


int main (void)
{
	printf("1. Mnist\n2. CIFAR-10\n3. Chinese\n4. CIFAR-100\nChoose the dataSet to run:");
	int cmd;
	scanf("%d", &cmd);
	if(cmd == 1)
		runMnist();
	else if(cmd == 2)
		runCifar10();
	else if(cmd == 3)
		runChinese();
	else if(cmd == 4)
		runCifar100();
	return EXIT_SUCCESS;
}

#ifdef linux
void runChinese() {
	const int nclasses = 10;
	/*state and cublas handle*/
	cublasHandle_t handle;
	init(handle);

	/*read the data from disk*/
	cuMatrixVector<double> trainX;
	cuMatrixVector<double> testX;
	cuMatrix<int>* trainY, *testY;
	readChineseData(trainX, testX, trainY, testY);

	Config::instance()->initPath("ChineseConfig.txt");

	/*build CNN net*/
	int ImgSize = trainX[0]->rows;
	int crop = Config::instance()->getCrop();

	int nsamples = trainX.size();
	std::vector<cuCvl> ConvLayers;
	std::vector<cuFll> HiddenLayers;
	cuSMR smr;
	int batch = Config::instance()->getBatchSize();
	double start, end;
	int cmd;
	cuInitDistortionMemery(batch, ImgSize - crop);
	printf(
			"1. random init weight\n2. Read weight from file\nChoose the way to init weight:");

	scanf("%d", &cmd);
	if (cmd == 1)
		cuConvNetInitPrarms(ConvLayers, HiddenLayers, smr, ImgSize - crop,
				nclasses);
	else if (cmd == 2)
		cuReadConvNet(ConvLayers, HiddenLayers, smr, ImgSize - crop,
				"checkPoint.txt", nclasses);

	cuInitCNNMemory(batch, trainX, testX, ConvLayers, HiddenLayers, smr,
			ImgSize - crop, nclasses);

	/*learning rate*/
	std::vector<double>nlrate;
	std::vector<double>nMomentum;
	std::vector<int>epoCount;
	double r = 0.05;
	double m = 0.90;
	int e = 50;
	for(int i = 0; i < 20; i++){
		nlrate.push_back(r);
		nMomentum.push_back(m);
		epoCount.push_back(e);
		r = r * 0.75;
		m = m + 0.005;
		if(m >= 1.0) m = 0.99;
	}
	start = clock();
	cuTrainNetwork(trainX, trainY, ConvLayers, HiddenLayers, smr, testX, testY, batch, ImgSize - crop, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	printf("trainning time %lf\n", (end - start) / CLOCKS_PER_SEC);
}
#else
void runChinese(){

}
#endif


void runCifar100(){
	/*state and cublas handle*/
	cublasHandle_t handle;
	init(handle);

	/*read the data from disk*/
	cuMatrixVector<double>trainX;
	cuMatrixVector<double>testX;
	cuMatrix<int>* trainY, *testY;

	Config::instance()->initPath("Cifar100Config.txt");
	read_CIFAR100_Data(trainX, testX, trainY, testY);
	preProcessing(trainX, testX);

	const int nclasses = Config::instance()->getSoftMax()[0]->m_numClasses;

	/*build CNN net*/
	int ImgSize = trainX[0]->rows;
	int crop = Config::instance()->getCrop();

	int nsamples = trainX.size();
	std::vector<cuCvl> ConvLayers;
	std::vector<cuFll> HiddenLayers;
	cuSMR smr;
	int batch = Config::instance()->getBatchSize();
	double start,end;
	int cmd;
	cuInitDistortionMemery(batch, ImgSize - crop);
	printf("1. random init weight\n2. Read weight from file\nChoose the way to init weight:");

	scanf("%d", &cmd);
	if(cmd == 1)
		cuConvNetInitPrarms(ConvLayers, HiddenLayers, smr, ImgSize - crop, nclasses);
	else if(cmd == 2)
		cuReadConvNet(ConvLayers, HiddenLayers, smr, ImgSize - crop, "checkPoint.txt", nclasses);

	cuInitCNNMemory(batch, trainX, testX, ConvLayers, HiddenLayers, smr, ImgSize - crop, nclasses);

	/*learning rate*/
	std::vector<double>nlrate;
	std::vector<double>nMomentum;
	std::vector<int>epoCount;
	double r = 0.05;
	double m = 0.90;
	int e = 50;
	for(int i = 0; i < 20; i++){
		nlrate.push_back(r);
		nMomentum.push_back(m);
		epoCount.push_back(e);
		r = r * 0.75;
		m = m + 0.005;
		if(m >= 1.0) m = 0.99;
	}
	start = clock();
	cuTrainNetwork(trainX, trainY, ConvLayers, HiddenLayers, smr, testX, testY, batch, ImgSize - crop, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();

	printf("trainning time %lf\n", (end - start) / CLOCKS_PER_SEC);
}
void runCifar10()
{
	/*state and cublas handle*/
	cublasHandle_t handle;
	init(handle);

	/*read the data from disk*/
	cuMatrixVector<double>trainX;
	cuMatrixVector<double>testX;
	cuMatrix<int>* trainY, *testY;

	Config::instance()->initPath("Cifar10Config.txt");
	read_CIFAR10_Data(trainX, testX, trainY, testY);
	preProcessing(trainX, testX);

	const int nclasses = Config::instance()->getSoftMax()[0]->m_numClasses;

	/*build CNN net*/
	int ImgSize = trainX[0]->rows;
	int crop = Config::instance()->getCrop();

	int nsamples = trainX.size();
	std::vector<cuCvl> ConvLayers;
	std::vector<cuFll> HiddenLayers;
	cuSMR smr;
	int batch = Config::instance()->getBatchSize();
	double start,end;
	int cmd;
	cuInitDistortionMemery(batch, ImgSize - crop);
	printf("1. random init weight\n2. Read weight from file\nChoose the way to init weight:");

	scanf("%d", &cmd);
	if(cmd == 1)
		cuConvNetInitPrarms(ConvLayers, HiddenLayers, smr, ImgSize - crop, nclasses);
	else if(cmd == 2)
		cuReadConvNet(ConvLayers, HiddenLayers, smr, ImgSize - crop, "checkPoint.txt", nclasses);

	cuInitCNNMemory(batch, trainX, testX, ConvLayers, HiddenLayers, smr, ImgSize - crop, nclasses);
	
	/*learning rate*/
	std::vector<double>nlrate;
	std::vector<double>nMomentum;
	std::vector<int>epoCount;
	double r = 0.05;
	double m = 0.90;
	int e = 50;
	for(int i = 0; i < 20; i++){
		nlrate.push_back(r);
		nMomentum.push_back(m);
		epoCount.push_back(e);
		r = r * 0.75;
		m = m + 0.005;
		if(m >= 1.0) m = 0.99;
	}
	start = clock();
	cuTrainNetwork(trainX, trainY, ConvLayers, HiddenLayers, smr, testX, testY, batch, ImgSize - crop, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();
	printf("trainning time %lf\n", (end - start) / CLOCKS_PER_SEC);
}

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
 	cuMatrix<int>* trainY, *testY;
	Config::instance()->initPath("MnistConfig.txt");
 	readMnistData(trainX, trainY, "mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", 60000, 1);
 	readMnistData(testX , testY,  "mnist/t10k-images.idx3-ubyte",  "mnist/t10k-labels.idx1-ubyte",  10000, 1);

 	/*build CNN net*/
 	int ImgSize = trainX[0]->rows;
	int crop    = Config::instance()->getCrop();

 	int nsamples = trainX.size();
 	std::vector<cuCvl> ConvLayers;
 	std::vector<cuFll> HiddenLayers;
 	cuSMR smr;
 	int batch = Config::instance()->getBatchSize();
	double start,end;
	int cmd;
	cuInitDistortionMemery(batch, ImgSize - crop);
	printf("1. random init weight\n2. Read weight from file\nChoose the way to init weight:");

	scanf("%d", &cmd);
 	if(cmd == 1)
		cuConvNetInitPrarms(ConvLayers, HiddenLayers, smr, ImgSize - crop, nclasses);
	else if(cmd == 2)
		cuReadConvNet(ConvLayers, HiddenLayers, smr, ImgSize - crop, "checkPoint.txt", nclasses);

	cuInitCNNMemory(batch, trainX, testX, ConvLayers,HiddenLayers, smr, ImgSize - crop, nclasses);



	/*learning rate*/
	std::vector<double>nlrate;
	std::vector<double>nMomentum;
	std::vector<int>epoCount;
	nlrate.push_back(0.05);   nMomentum.push_back(0.90);  epoCount.push_back(50);
	nlrate.push_back(0.04);   nMomentum.push_back(0.91);  epoCount.push_back(50);
	nlrate.push_back(0.03);   nMomentum.push_back(0.92);  epoCount.push_back(50);
	nlrate.push_back(0.02);   nMomentum.push_back(0.93);  epoCount.push_back(50);
	nlrate.push_back(0.01);   nMomentum.push_back(0.94);  epoCount.push_back(50);
	nlrate.push_back(0.008);  nMomentum.push_back(0.942); epoCount.push_back(50);
	nlrate.push_back(0.006);  nMomentum.push_back(0.944); epoCount.push_back(50);
	nlrate.push_back(0.001);  nMomentum.push_back(0.95);  epoCount.push_back(50);
	nlrate.push_back(0.0009); nMomentum.push_back(0.96);  epoCount.push_back(50);
	nlrate.push_back(0.0008); nMomentum.push_back(0.97);  epoCount.push_back(50);
	nlrate.push_back(0.0007); nMomentum.push_back(0.995); epoCount.push_back(50);
	nlrate.push_back(0.0006); nMomentum.push_back(0.90);  epoCount.push_back(50);

	start = clock();
	cuTrainNetwork(trainX, trainY, ConvLayers, HiddenLayers, smr, testX, testY, batch, ImgSize - crop, nclasses, nlrate, nMomentum, epoCount, handle);
	end = clock();
	printf("trainning time %lf\n", (end - start) / CLOCKS_PER_SEC);
}

