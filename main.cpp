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
void cuVoteMnist();
bool init(cublasHandle_t& handle);


int main (void)
{
	printf("1. MNIST\n2. CIFAR-10\n3. CHINESE\n4. CIFAR-100\n5. VOTE MNIST\nChoose the dataSet to run:");
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
	else if(cmd == 5)
		cuVoteMnist();
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
	int e = 10;
	for(int i = 0; i < 100; i++){
		nlrate.push_back(r);
		nMomentum.push_back(m);
		epoCount.push_back(e);
		r = r * 0.95;
		m = m + 0.001;
		if(m >= 1.0) break;
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
	int e = 10;
	for(int i = 0; i < 100; i++){
		nlrate.push_back(r);
		nMomentum.push_back(m);
		epoCount.push_back(e);
		r = r * 0.95;
		m = m + 0.001;
		if(m >= 1.0) break;
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



void cuVoteMnist()
{
	const int nclasses = 10;

	/*state and cublas handle*/
	cublasHandle_t handle;
	init(handle);

	/*read the data from disk*/
	cuMatrixVector<double>trainX;
	cuMatrixVector<double>testX;
	cuMatrix<int>* trainY, *testY;

	readMnistData(trainX, trainY, "mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", 60000, 1);
 	readMnistData(testX , testY,  "mnist/t10k-images.idx3-ubyte",  "mnist/t10k-labels.idx1-ubyte",  10000, 1);

	int ImgSize = trainX[0]->rows;

	char* path[] = {"mnist_result_cfm_5/1/checkPoint.txt",
		"mnist_result_cfm_5/2/checkPoint.txt",
		"mnist_result_cfm_5/3/checkPoint.txt",
		"mnist_result_cfm_5/4/checkPoint.txt",
		"mnist_result_cfm_5/5/checkPoint.txt",
		"mnist_result_cfm_5/6/checkPoint.txt",
		"mnist_result_cfm_5/7/checkPoint.txt",
		"mnist_result_cfm_5/8/checkPoint.txt",
		"mnist_result_cfm_5/9/checkPoint.txt"};

	char* initPath[] = {"mnist_result_cfm_4/1/MnistConfig.txt",
		"mnist_result_cfm_5/2/MnistConfig.txt",
		"mnist_result_cfm_5/3/MnistConfig.txt",
		"mnist_result_cfm_5/4/MnistConfig.txt",
		"mnist_result_cfm_5/5/MnistConfig.txt",
		"mnist_result_cfm_5/6/MnistConfig.txt",
		"mnist_result_cfm_5/7/MnistConfig.txt",
		"mnist_result_cfm_5/8/MnistConfig.txt",
		"mnist_result_cfm_5/9/MnistConfig.txt"};

	/*build CNN net*/
	/*
	char* path[] = {"mnist_result_cfm_2/1_9967/checkPoint.txt",
		"mnist_result_cfm_2/2_9964/checkPoint.txt",
		"mnist_result_cfm_2/3_9966/checkPoint.txt",
		"mnist_result_cfm_2/4_9970/checkPoint.txt",
		"mnist_result_cfm_2/5_9967/checkPoint.txt",
		"mnist_result_cfm_2/6_9963/checkPoint.txt",
		"mnist_result_cfm_2/7_9965/checkPoint.txt",
		"mnist_result_cfm_2/8_9963/checkPoint.txt",
		"mnist_result_cfm_2/9_9964/checkPoint.txt"};

	char* initPath[] = {"mnist_result_cfm_2/1_9967/MnistConfig.txt",
		"mnist_result_cfm_2/2_9964/MnistConfig.txt",
		"mnist_result_cfm_2/3_9966/MnistConfig.txt",
		"mnist_result_cfm_2/4_9970/MnistConfig.txt",
		"mnist_result_cfm_2/5_9967/MnistConfig.txt",
		"mnist_result_cfm_2/6_9963/MnistConfig.txt",
		"mnist_result_cfm_2/7_9965/MnistConfig.txt",
		"mnist_result_cfm_2/8_9963/MnistConfig.txt",
		"mnist_result_cfm_2/9_9964/MnistConfig.txt"};
		*/
	/*
	ncfm
	char* path[] = {"mnist_result/1_9973/checkPoint.txt",
	"mnist_result/2_9972/checkPoint.txt",
	"mnist_result/3_9969/checkPoint.txt",
	"mnist_result/4_9968/checkPoint.txt",
	"mnist_result/5_9974/checkPoint.txt",
	"mnist_result/6_9973/checkPoint.txt",
	"mnist_result/7_9968/checkPoint.txt",
	"mnist_result/8_9968/checkPoint.txt",
	"mnist_result/9_9968/checkPoint.txt",
	"mnist_result/10_9976/checkPoint.txt",
	"mnist_result/11_9969/checkPoint.txt",
	"mnist_result/12_9968/checkPoint.txt",
	"mnist_result/13_9973/checkPoint.txt"};

	char* initPath[] = {"mnist_result/1_9973/MnistConfig.txt",
	"mnist_result/2_9972/MnistConfig.txt",
	"mnist_result/3_9969/MnistConfig.txt",
	"mnist_result/4_9968/MnistConfig.txt",
	"mnist_result/5_9974/MnistConfig.txt",
	"mnist_result/6_9973/MnistConfig.txt",
	"mnist_result/7_9968/MnistConfig.txt",
	"mnist_result/8_9968/MnistConfig.txt",
	"mnist_result/9_9968/MnistConfig.txt",
	"mnist_result/10_9976/MnistConfig.txt",
	"mnist_result/11_9969/MnistConfig.txt",
	"mnist_result/12_9968/MnistConfig.txt",
	"mnist_result/13_9973/MnistConfig.txt"};
	*/

	int numPath = sizeof(path) / sizeof(char*);
	int * mark  = new int[1 << numPath];
	memset(mark, 0, sizeof(int) * (1 << numPath));
	std::vector<int>vCorrect;
	std::vector<cuMatrix<int>*>vPredict;

	for(int i = 0; i < numPath; i++)
	{
		Config::instance()->initPath(initPath[i]);

		int ImgSize = trainX[0]->rows;
		int crop    = Config::instance()->getCrop();

		int nsamples = trainX.size();
		std::vector<cuCvl> ConvLayers;
		std::vector<cuFll> HiddenLayers;
		cuSMR smr;
		int batch = Config::instance()->getBatchSize();

		vPredict.push_back(new cuMatrix<int>(testY->getLen(), nclasses, 1));
		cuReadConvNet(ConvLayers, HiddenLayers, smr, ImgSize - crop, path[i], nclasses);
		cuInitCNNMemory(batch, trainX, testX, ConvLayers, HiddenLayers, smr, ImgSize, nclasses);
		int cur = voteTestDate(
			ConvLayers,
			HiddenLayers,
			smr,
			testX,
			testY,
			vPredict[i],
			batch,
			ImgSize,
			nclasses,
			handle);
		cuFreeCNNMemory(batch, trainX, testX, ConvLayers,HiddenLayers, smr);
		cuFreeConvNet(ConvLayers, HiddenLayers, smr);
		vCorrect.push_back(cur);
		Config::instance()->clear();
		printf("%d %s\n", cur, path[i]);
	}
	
	int max = -1;
	int val = 1;
	int cur;
	cuMatrix<int>* voteSum = new cuMatrix<int>(testY->getLen(), nclasses, 1);
	cuMatrix<int>* correct = new cuMatrix<int>(1, 1, 1);

	for(int m = (1 << numPath) - 1; m >= 1; m--)
	{
		int t = 0;
		for(int i = 0; i < numPath; i++){
			if(m & (1 << i)){
				t++;
			}
		}
		if(t != 5) continue;
		voteSum->gpuClear();
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
				correct->gpuClear();
				cur = cuVoteAdd(voteSum, vPredict[i], testY, correct, nclasses);
				mark[v] = cur;
				printf("%d %d %s\n", vCorrect[i], cur, path[i]);
			}
		}
		if(cur >= max)
		{
			max = cur;
			val = m;
		}

		printf("m = %d val = %d max = %d \n\n",m, val, max);
	}

	voteSum->gpuClear();
	int m = val;
	for(int i = numPath - 1; i >= 0; i--)
	{
		if(!(m & (1 << i)))continue;
		correct->gpuClear();
		cur = cuVoteAdd(voteSum, vPredict[i], testY, correct, nclasses);
		printf("%d %d %s\n", vCorrect[i], cur, path[i]);
	}
}