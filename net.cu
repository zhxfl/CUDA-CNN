#include "net.cuh"
#include "opencv2/opencv.hpp"
#include "cuMatrix.h"
#include <cuda_runtime.h>
#include "util.h"
#include <time.h>
#include "cuTrasformation.cuh"
#include "Config.h"
#include "cuMatrixVector.h"
#include <helper_functions.h>
#include <helper_cuda.h>

cuMatrixVector<double>* cu_distortion_vector;

std::vector<int> cuPoolOutputSize;
std::vector<int> cuConvOutputSize;
std::vector<int> cuKernelScan;

//�������
std::vector<cuMatrix<int>*>cuPointX;
std::vector<cuMatrix<int>*>cuPointY;
std::vector<cuMatrix<double>*>cuConv;
std::vector<cuMatrix<double>*>cuPool;

cuMatrix<double>* cuPoolToFlActi;

//���ز�����
std::vector<cuMatrix<double>*>cuHiddenActi;

//�ع������
cuMatrix<double>* cuSoftMaxP;
cuMatrix<double>* cuGroundTruth;

//���򴫵�
cuMatrix<double>* cuSoftMaxDelta;
std::vector<cuMatrix<double>*>cuHiddenDelta;
std::vector<cuMatrix<double>*>cuConvDelta;
std::vector<cuMatrix<double>*>cuPoolDelta;
std::vector<cuMatrix<double>*>cuPoolDeltaAndBorder;
std::vector<cuMatrix<double>*>cuConvLayerWgradTmp;

cuMatrix<double>* cuPoolToFlDelta;

//��
cuMatrix<double>* cu_v_smr_w;
cuMatrix<double>* cu_v_smr_b;
std::vector<cuMatrix<double>*>cu_v_hl_w;
std::vector<cuMatrix<double>*>cu_v_hl_b;
std::vector<std::vector<cuMatrix<double>*> >cu_v_cvl_w;
std::vector<std::vector<cuMatrix<double>*> >cu_v_cvl_b;
int cuCurCorrect;
cuMatrix<int>*cuCorrect = NULL;
cuMatrix<int>*cuVote = NULL;

void outputMatrix(cuMatrix<double>* m);
/*
	�����ܣ������ж���ˣ���GPU������ʱ��
			  �ڴ�Ƚ�ɢ�ң����ｲ��Щ�˵Ĳ�����֯��
			  ��ά���鷽�㲢������
*/
void cuConvLayer::init()
{
	w = new cuMatrixVector<double>();
	for(int i = 0; i < layer.size(); i++){
		w->push_back(layer[i].W);
	}
	w->toGpu();


	b = new cuMatrixVector<double>();
	for(int i = 0; i < layer.size(); i++){
		b->push_back(layer[i].b);
	}
	b->toGpu();


	wgrad = new cuMatrixVector<double>();
	for(int i = 0; i < layer.size(); i++){
		wgrad->push_back(layer[i].Wgrad);
	}
	wgrad->toGpu();


	bgrad = new cuMatrixVector<double>();
	for(int i = 0; i < layer.size(); i++){
		bgrad->push_back(layer[i].bgrad);
	}
	bgrad->toGpu();
}

void createGaussian(double* gaussian, double dElasticSigma1, double dElasticSigma2, int rows, int cols, int channels, double epsilon)
{
	int iiMidr = rows >> 1;
	int iiMidc = cols >> 1;
	
	double _max = -1.0;
	for(int row = 0; row < rows; row++)
	{
		for(int col = 0; col < cols; col++)
		{
			double val1 = 1.0 / (dElasticSigma1 * dElasticSigma2 * 2.0 * 3.1415926535897932384626433832795);
			double val2 = (row-iiMidr)*(row-iiMidr) / (dElasticSigma1 * dElasticSigma1) + (col-iiMidc)*(col-iiMidc) / (dElasticSigma2 * dElasticSigma2) 
				+ 2.0 * (row - iiMidr) * (col - iiMidc) / (dElasticSigma1 * dElasticSigma2);
			gaussian[row * cols + col] = val1 * exp(-1.0 * val2);
			if(_max < gaussian[row * cols + col])
			{
				_max = gaussian[row * cols + col];
			}
		}
	}
	for(int row = 0; row < rows; row++)
	{
		for(int col = 0; col < cols; col++)
		{
			gaussian[row * cols + col] /= _max;
			gaussian[row * cols + col] *= epsilon;
			//printf("%lf ", gaussian[row * cols + col]);
		}
		//printf("\n");
	}
}

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
	return 0.0;
}

__device__ double d_dnonLinearity(double val,int NONLIN){
	if(NONLIN == NL_RELU)
	{
		if(val > 0.0) return 1.0;
		else return val;
	}
	else if(NONLIN == NL_TANH)
	{
		double res = 1.7159;
		double temp = val * val / 1.7159;
		res = (res - temp) * 2.0 / 3.0;
		return res;
	}
}

void dropDelta(cuMatrix<double>* M, double cuDropProb)
{
	for(int c = 0; c < M->channels; c++){
		cv::Mat ran = cv::Mat::zeros(M->rows, M->cols, CV_64FC1);
		cv::theRNG().state = clock();
		randu(ran, cv::Scalar(0), cv::Scalar(1.0));
		for(int i = 0; i < M->rows; i++){
			for(int j = 0; j < M->cols; j++){
				double r = ran.at<double>(i, j);
				if(r < cuDropProb)
					M->set(i, j, c, 0.0);
				else 
					M->set(i, j, c, 1.0);
				//printf("%lf ", fll.M->get(i,j));
			}//printf("\n");
		}
	}
	M->toGpu();
}

/*
�����ܣ���������ʼ��
��    ��
ConvK &convk  ����
int width     ��С
*/

void weightRandomInit(cuConvK &convk, int width){
	double epsilon = 0.1f;
	convk.W = new cuMatrix<double>(width, width, Config::instance()->getChannels());

	//printf("%f %f\n",r1, r2);
	for(int c = 0; c < Config::instance()->getChannels(); c++)
	{
		double r1 = 0.5 + 4.0 * (rand()) / RAND_MAX;
		double r2 = 0.5 + 4.0 * (rand()) / RAND_MAX;
		createGaussian(convk.W->hostData + c * convk.W->getArea(), r1,r2,
			width, width, 
			Config::instance()->getChannels(), 
			epsilon * 0.5 + epsilon * rand() / RAND_MAX);
	}

// 	for(int i = 0; i<convk.W->rows; i++){
// 		for(int j=0; j<convk.W->cols; j++){
// 			convk.W->set(i, j, 1.0 * rand() / RAND_MAX * 2.0 * epsilon - epsilon);
// 		}
// 	}
	convk.W->toGpu();
	convk.b = new cuMatrix<double>(1, 1, Config::instance()->getChannels());
	convk.Wgrad = new cuMatrix<double>(width, width, Config::instance()->getChannels());
	convk.bgrad = new cuMatrix<double>(1, 1, Config::instance()->getChannels());
}


/*
�����ܣ����ز�����ʼ��
��    ��
fll &fll ���ز�
int inputsize  ÿ����Ԫ������һ����Ԫ����
int hiddensize �������Ԫ����
*/

void weightRandomInit(cuFll &fll, int inputsize, int hiddensize, double dropRate){
	double epsilon = sqrt((double)6) / sqrt((double)(hiddensize + inputsize + 1));
	fll.W = new cuMatrix<double>(hiddensize, inputsize, 1);
	fll.dropW = new cuMatrix<double>(hiddensize, inputsize, 1);
	fll.afterDropW = new cuMatrix<double>(hiddensize, inputsize, 1);

	//double r1 = 2.0 + 4.0 * (rand()) / RAND_MAX;
	//double r2 = 2.0 + 4.0 * (rand()) / RAND_MAX;
	//createGaussian(fll.W->hostData, r1, r2, fll.W->rows, fll.W->cols, epsilon * 0.5 + epsilon * rand() / RAND_MAX);
	for(int c=0; c < fll.W->channels; c++){
		for(int i=0; i < fll.W->rows; i++){
			for(int j=0; j< fll.W->cols; j++){
				fll.W->set(i,j, c, 1.0 * rand() / RAND_MAX *  2.0 * epsilon - epsilon);
				//fll.W->set(i, j, c, 0.001);
			}
		}
	}

	//dropDelta(fll.dropW, dropRate);
	fll.W->toGpu();
	fll.b = new cuMatrix<double>(hiddensize, 1, 1);
	fll.Wgrad = new cuMatrix<double>(hiddensize, inputsize, 1);
	fll.bgrad = new cuMatrix<double>(hiddensize, 1, 1);
}

/*
�����ܣ�softMax����Ԫ��ʼ��
��    ��
SMR &smr      softMax��
int nclasses  ���������
int nfeatures ����
*/
void weightRandomInit(cuSMR &smr, int nclasses, int nfeatures){
	double epsilon = 0.01f;
	smr.Weight = new cuMatrix<double>(nclasses, nfeatures, 1);
	for(int c = 0; c < smr.Weight->channels; c++){
		for(int i = 0; i<smr.Weight->rows; i++){
			for(int j=0; j<smr.Weight->cols; j++){
				smr.Weight->set(i,j, c, 1.0 * rand() / RAND_MAX *  2.0 * epsilon - epsilon);     
			}
		}
	}
	
	smr.Weight->toGpu();

	smr.b = new cuMatrix<double>(nclasses, 1, 1);
	smr.cost = new cuMatrix<double>(1, 1, 1);
	smr.Wgrad = new cuMatrix<double>(nclasses, nfeatures, 1);
	smr.bgrad = new cuMatrix<double>(nclasses, 1, 1);
}

void cuConvNetInitPrarms(std::vector<cuCvl> &ConvLayers,
	std::vector<cuFll> &HiddenLayers,
	cuSMR &smr,
	int imgDim,
	int nsamples, 
	int nclasses)
{
	srand(clock());
	// Init Conv layers
	for(int i=0; i < Config::instance()->getConv().size(); i++){
		cuCvl tpcvl;
		int kernelSize = Config::instance()->getConv()[i]->m_kernelSize;
		for(int j = 0; j < Config::instance()->getConv()[i]->m_amount; j++){
			cuConvK tmpConvK;
			weightRandomInit(tmpConvK, kernelSize);
			tpcvl.layer.push_back(tmpConvK);
		}
		ConvLayers.push_back(tpcvl);
	}
	for(int i = 0; i < Config::instance()->getConv().size(); i++){
		ConvLayers[i].init();
	}

	// Init Hidden layers
	int outDim = imgDim;
	for(int i=0; i<Config::instance()->getConv().size(); i++){
		outDim = outDim - Config::instance()->getConv()[i]->m_kernelSize + 1;
		outDim = (outDim + Config::instance()->getConv()[i]->m_poolingDim - 1) / Config::instance()->getConv()[i]->m_poolingDim;
	}
	int hiddenfeatures = pow((double)outDim, 2.0);
	for(int i=0; i<ConvLayers.size(); i++){
		hiddenfeatures *= ConvLayers[i].layer.size();
	}
	cuFll tpfll;
	weightRandomInit(tpfll, hiddenfeatures * Config::instance()->getChannels(), Config::instance()->getFC()[0]->m_numHiddenNeurons, 
		Config::instance()->getFC()[0]->m_dropoutRate);
	HiddenLayers.push_back(tpfll);
	for(int i=1; i < Config::instance()->getFC().size(); i++){
		cuFll tpfll2;
		weightRandomInit(tpfll2, Config::instance()->getFC()[i - 1]->m_numHiddenNeurons,
			Config::instance()->getFC()[i]->m_numHiddenNeurons, Config::instance()->getFC()[i]->m_dropoutRate);
		HiddenLayers.push_back(tpfll2);
	}
	// Init Softmax layer
	weightRandomInit(smr, nclasses, Config::instance()->getFC()[Config::instance()->getFC().size() - 1]->m_numHiddenNeurons);
}

void saveWeight(cuConvK &convk, int width, FILE*pOut)
{
	convk.W->toCpu();
	convk.b->toCpu();
	for(int c = 0; c < convk.W->channels; c++){
		for(int i = 0; i<convk.W->rows; i++){
			for(int j=0; j<convk.W->cols; j++){
				fprintf(pOut, "%lf ",convk.W->get(i,j, c));      
			}
		}
	}
	for(int c = 0; c < convk.b->channels; c++)
		fprintf(pOut, "%lf ", convk.b->get(0,0, c));
}

void saveWeight(cuFll &fll, int inputsize, int hiddensize, FILE*pOut){
	fll.W->toCpu();
	fll.b->toCpu();
	for(int c = 0; c < fll.W->channels; c++){
		for(int i=0; i<fll.W->rows; i++){
			for(int j=0; j<fll.W->cols; j++){
				fprintf(pOut, "%lf ", fll.W->get(i,j,c));
			}
		}
	}
	
	for(int c = 0; c < fll.b->channels; c++){
		for(int i = 0; i < fll.b->rows; i++){
			for(int j = 0; j < fll.b->cols;  j++){
				fprintf(pOut, "%lf ", fll.b->get(i,j, c));
			}
		}
	}
}

void saveWeight(cuSMR &smr, int nclasses, int nfeatures, FILE* pOut){
	smr.Weight->toCpu();
	smr.b->toCpu();
	for(int c = 0; c < smr.Weight->channels; c++){
		for(int i = 0; i<smr.Weight->rows; i++){
			for(int j=0; j<smr.Weight->cols; j++){
				fprintf(pOut, "%lf ", smr.Weight->get(i,j,c)); 
			}
		}
	}

	for(int c = 0; c < smr.b->channels; c++){
		for(int i = 0; i < smr.b->rows; i++){
			for(int j = 0; j < smr.b->cols;  j++){
				fprintf(pOut, "%lf ", smr.b->get(i,j,c));
			}
		}
	}
	
}

void cuSaveConvNet(std::vector<cuCvl> &ConvLayers,
	std::vector<cuFll> &HiddenLayers,
	cuSMR &smr,
	int imgDim,
	int nclasses,
	int nsamples)
{	
	FILE *pOut = fopen("checkPoint.txt", "w");
	// Init Conv layers
	for(int i=0; i < Config::instance()->getConv().size(); i++){
		cuCvl tpcvl = ConvLayers[i];
		for(int j=0; j < Config::instance()->getConv()[i]->m_amount; j++){
			cuConvK tmpConvK = tpcvl.layer[j];
			saveWeight(tmpConvK, Config::instance()->getConv()[i]->m_kernelSize, pOut);
		}
	}

	// Init Hidden layers
	int outDim = imgDim;
	for(int i=0; i < Config::instance()->getConv().size(); i++){
		outDim = outDim - Config::instance()->getConv()[i]->m_kernelSize + 1;
		outDim = (outDim + Config::instance()->getConv()[i]->m_poolingDim - 1) / Config::instance()->getConv()[i]->m_poolingDim;
	}

	int hiddenfeatures = pow((double)outDim, (double)2);
	for(int i=0; i<ConvLayers.size(); i++){
		hiddenfeatures *= ConvLayers[i].layer.size();
	}

	cuFll tpfll = HiddenLayers[0];
	saveWeight(tpfll, hiddenfeatures * Config::instance()->getChannels(), Config::instance()->getFC()[0]->m_numHiddenNeurons, pOut);
	for(int i=1; i < Config::instance()->getFC().size(); i++){
		cuFll tpfll2 = HiddenLayers[i];
		saveWeight(tpfll2, Config::instance()->getFC()[i - 1]->m_numHiddenNeurons, 
			Config::instance()->getFC()[i]->m_numHiddenNeurons, pOut);
	}
	// Init Softmax layer
	saveWeight(smr, nclasses, Config::instance()->getFC()[Config::instance()->getFC().size() - 1]->m_numHiddenNeurons, pOut);
	fclose(pOut);
};

void readWeight(cuConvK &convk, int width, FILE*pIn)
{
	convk.W = new cuMatrix<double>(width, width, Config::instance()->getChannels()); 
	double val = 0.0;
	for(int c = 0; c < convk.W->channels; c++){
		for(int i = 0; i<convk.W->rows; i++){
			for(int j=0; j<convk.W->cols; j++){
				fscanf(pIn, "%lf", &val); 
				convk.W->set(i,j,c,val);
			}
		}
	}

	convk.b = new cuMatrix<double>(1, 1, Config::instance()->getChannels());
	for(int c = 0; c < convk.b->channels; c++)
	{
		fscanf(pIn, "%lf ", &val);
		convk.b->set(0,0,c,val);
	}
	convk.Wgrad = new cuMatrix<double>(width, width, Config::instance()->getChannels());
	convk.bgrad = new cuMatrix<double>(1, 1, Config::instance()->getChannels());
	convk.W->toGpu();
	convk.b->toGpu();
}

void readWeight(cuFll &fll, int inputsize, int hiddensize, FILE*pIn, double dropRate){
	double val = 0.0;
	fll.W = new cuMatrix<double>(hiddensize, inputsize, 1);
	for(int c = 0; c < fll.W->channels; c++){
		for(int i=0; i<fll.W->rows; i++){
			for(int j=0; j<fll.W->cols; j++){
				fscanf(pIn, "%lf", &val);
				fll.W->set(i,j,c,val);
			}
		}
	}


	fll.dropW = new cuMatrix<double>(hiddensize, inputsize, 1);
	fll.afterDropW = new cuMatrix<double>(hiddensize, inputsize, 1);
	dropDelta(fll.dropW, dropRate);


	fll.b = new cuMatrix<double>(hiddensize, 1, 1);
	for(int i = 0; i < fll.b->rows; i++){
		for(int j = 0; j < fll.b->cols; j++){
			fscanf(pIn, "%lf", &val);
			fll.b->set(i,j,0,val);
		}
	}
	
	fll.Wgrad = new cuMatrix<double>(hiddensize, inputsize, 1);
	fll.bgrad = new cuMatrix<double>(hiddensize, 1, 1);
	fll.W->toGpu();
	fll.b->toGpu();
}

void readWeight(cuSMR &smr, int nclasses, int nfeatures, FILE* pIn){
	smr.Weight = new cuMatrix<double>(nclasses, nfeatures, 1);
	double val = 0.0;
	for(int i = 0; i<smr.Weight->rows; i++){
		for(int j=0; j<smr.Weight->cols; j++){
			fscanf(pIn, "%lf", &val);
			smr.Weight->set(i,j,0,val);
		}
	}
	smr.b = new cuMatrix<double>(nclasses, 1, 1);
	for(int i = 0; i < smr.b->rows; i++){
		for(int j = 0; j < smr.b->cols; j++){
			fscanf(pIn, "%lf ", &val);
			smr.b->set(i,j,0, val);
		}
	}
	smr.cost = new cuMatrix<double>(1,1,1);
	smr.Wgrad = new cuMatrix<double>(nclasses, nfeatures, 1);
	smr.bgrad = new cuMatrix<double>(nclasses, 1, 1);
	smr.Weight->toGpu();
	smr.b->toGpu();
}

void cuFreeConvNet(std::vector<cuCvl> &ConvLayers,
	std::vector<cuFll> &HiddenLayers,
	cuSMR &smr)
{
	for(int cl = 0; cl <ConvLayers.size(); cl++)
	{
		ConvLayers[cl].clear();
	}
	ConvLayers.clear();
	for(int hl = 0; hl < HiddenLayers.size(); hl++)
	{
		HiddenLayers[hl].clear();
	}
	HiddenLayers.clear();
	smr.clear();
}

void cuReadConvNet(std::vector<cuCvl> &ConvLayers,
	std::vector<cuFll> &HiddenLayers,
	cuSMR &smr, int imgDim, int nsamples, char* path,
	int nclasses)
{	
	FILE *pIn = fopen(path, "r");
	// Init Conv layers
	// ������Ȼû��ǳ������й¶�����⣬�������ڴ��������Ȼ��������
	// cuConvk��һ����ʱ����,ϵͳ���ͷ���(new������û���ͷ�)
	// ����push_back�½�һ���������õ���֮ǰû���ͷŵĿռ䣬���Ըպ�û��й¶
	for(int i=0; i < Config::instance()->getConv().size(); i++){
		cuCvl tpcvl;
		for(int j=0; j < Config::instance()->getConv()[i]->m_amount; j++){
			cuConvK tmpConvK;
			readWeight(tmpConvK, Config::instance()->getConv()[i]->m_kernelSize, pIn);
			tpcvl.layer.push_back(tmpConvK);
		}
		ConvLayers.push_back(tpcvl);
	}

	for(int i = 0; i < Config::instance()->getConv().size(); i++)
	{
		ConvLayers[i].init();
	}
	

	//Init Hidden layers
	int outDim = imgDim;
	for(int i=0; i < Config::instance()->getConv().size(); i++){
		outDim = outDim - Config::instance()->getConv()[i]->m_kernelSize + 1;
		outDim = (outDim + Config::instance()->getConv()[i]->m_poolingDim - 1) / Config::instance()->getConv()[i]->m_poolingDim;
	}

	int hiddenfeatures = pow((double)outDim, (double)2);
	for(int i=0; i<ConvLayers.size(); i++){
		hiddenfeatures *= ConvLayers[i].layer.size();
	}

	cuFll tpfll;
	readWeight(tpfll, hiddenfeatures * Config::instance()->getChannels(), Config::instance()->getFC()[0]->m_numHiddenNeurons,
		pIn, Config::instance()->getFC()[0]->m_dropoutRate);
	HiddenLayers.push_back(tpfll);
	for(int i=1; i < Config::instance()->getFC().size(); i++){
		cuFll tpfll2;
		readWeight(tpfll2, Config::instance()->getFC()[i - 1]->m_numHiddenNeurons, 
			Config::instance()->getFC()[i]->m_numHiddenNeurons, pIn, Config::instance()->getFC()[i]->m_dropoutRate);
		HiddenLayers.push_back(tpfll2);
	}
	//Init Softmax layer
	readWeight(smr, nclasses, Config::instance()->getFC()[Config::instance()->getFC().size() - 1]->m_numHiddenNeurons, pIn);
	fclose(pIn);
};

void cuInitCNNMemory(
	int batch,
	cuMatrixVector<double>& trainX, 
	cuMatrixVector<double>& testX,
	std::vector<cuCvl>& ConvLayers,
	std::vector<cuFll>& HiddenLayers,
	cuSMR& smr,
	int ImgSize,
	int nclasses)
{
	cudaError_t cudaStat;
	//////////////
	//��������
	//////////////
	int curSize = ImgSize;
	int curKernelAmount = 1;
	for(int i = 0; i < ConvLayers.size(); i++){

		curSize = curSize - Config::instance()->getConv()[i]->m_kernelSize + 1;

		cuConvOutputSize.push_back(curSize);
		cuPoolOutputSize.push_back((curSize + Config::instance()->getConv()[i]->m_poolingDim - 1) /
			Config::instance()->getConv()[i]->m_poolingDim);

		curKernelAmount = curKernelAmount * Config::instance()->getConv()[i]->m_amount;
		cuKernelScan.push_back(curKernelAmount);

		cuConv.push_back  (new cuMatrix<double>(batch, curKernelAmount * curSize * curSize, Config::instance()->getChannels()));
		curSize = (curSize + Config::instance()->getConv()[i]->m_poolingDim - 1) / Config::instance()->getConv()[i]->m_poolingDim;
		cuPool.push_back  (new cuMatrix<double>(batch, curKernelAmount * curSize * curSize, Config::instance()->getChannels()));
		cuPointX.push_back(new cuMatrix<int>(batch, curKernelAmount * curSize * curSize, Config::instance()->getChannels()));
		cuPointY.push_back(new cuMatrix<int>(batch, curKernelAmount * curSize * curSize, Config::instance()->getChannels()));
	}

	////////////////////////////////////////////////////////////////////
	//translate the last cuPool from cuMatrix(batch, size, channels)
	//to cuPoolToFlActi cuMatrix(batch, size * channels) 
	//prepare for matrixMul
	////////////////////////////////////////////////////////////////////
	cuPoolToFlActi = new cuMatrix<double>(cuPool[cuPool.size() - 1]->rows, 
		cuPool[cuPool.size() - 1]->cols * cuPool[cuPool.size() - 1]->channels,
		1);

	//////////////
	//���ز�����
	//////////////
	for(int i = 0; i < HiddenLayers.size(); i++)
	{
		cuHiddenActi.push_back(new cuMatrix<double>(batch, Config::instance()->getFC()[i]->m_numHiddenNeurons, 1));
	}

	//////////////
	//�ع������
	//////////////
	cuSoftMaxP    = new cuMatrix<double>(batch, nclasses, 1);
	cuGroundTruth = new cuMatrix<double>(batch, nclasses, 1);

	/////////////
	//���򴫵�
	/////////////
	cuSoftMaxDelta = new cuMatrix<double>(batch, nclasses, 1);
	for(int i = 0; i < HiddenLayers.size(); i++)
	{
		cuHiddenDelta.push_back(new cuMatrix<double>(batch, Config::instance()->getFC()[i]->m_numHiddenNeurons, 1));
	}

	//////////
	//����
	//////////
	for(int i = 0; i < cuPool.size(); i++)
	{
		cuPoolDelta.push_back(new cuMatrix<double>(cuPool[i]->rows, cuPool[i]->cols, Config::instance()->getChannels()));
	}
	for(int i = 0; i < cuConv.size(); i++)
	{
		cuConvDelta.push_back(new cuMatrix<double>(cuConv[i]->rows, cuConv[i]->cols, Config::instance()->getChannels()));
	}
	for(int cl = ConvLayers.size() - 1; cl > 0; cl--)
	{
		int size = cuConvOutputSize[cl] + Config::instance()->getConv()[cl]->m_kernelSize - 1;
		cuPoolDeltaAndBorder.push_back(new cuMatrix<double>(batch, cuKernelScan[cl] * size * size, Config::instance()->getChannels()));
	}
	for(int cl = 0; cl < ConvLayers.size(); cl++)
	{
		cuConvLayerWgradTmp.push_back(new cuMatrix<double>(batch,
			cuKernelScan[cl] * Config::instance()->getConv()[cl]->m_kernelSize * Config::instance()->getConv()[cl]->m_kernelSize,
			Config::instance()->getChannels()));
	}

	////////////////////////////////////////////////////////////////////
	//translate the cuPoolToFlDelta from cuMatrix(batch, size * channels, 1)
	//to cuPoolDelta  cuMatrix(batch, size, channels) 
	////////////////////////////////////////////////////////////////////
	cuPoolToFlDelta = new cuMatrix<double>(cuPoolDelta[cuPoolDelta.size() - 1]->rows,
		cuPoolDelta[cuPoolDelta.size() - 1]->cols * Config::instance()->getChannels(),
		1);

	//����
	cu_v_smr_w = new cuMatrix<double>(smr.Weight->rows, smr.Weight->cols, smr.Weight->channels);
	cu_v_smr_b = new cuMatrix<double>(smr.b->rows, smr.b->cols, smr.b->channels);
	for(int i = 0; i < HiddenLayers.size(); i++)
	{
		cu_v_hl_w.push_back(new cuMatrix<double>(HiddenLayers[i].W->rows, HiddenLayers[i].W->cols, HiddenLayers[i].W->channels));
		cu_v_hl_b.push_back(new cuMatrix<double>(HiddenLayers[i].b->rows, HiddenLayers[i].b->cols, HiddenLayers[i].b->channels));
	}

	for(int cl = 0; cl < ConvLayers.size(); cl++)
	{
		std::vector<cuMatrix<double>*>tmpVecW;
		std::vector<cuMatrix<double>*>tmpVecb;
		for(int i = 0; i < ConvLayers[cl].layer.size(); i++)
		{
			cuMatrix<double>* tmpW = new cuMatrix<double>(ConvLayers[cl].layer[i].W->rows, 
				ConvLayers[cl].layer[i].W->cols,
				ConvLayers[cl].layer[i].W->channels);
			cuMatrix<double>* tmpb = new cuMatrix<double>(ConvLayers[cl].layer[i].b->rows,
				ConvLayers[cl].layer[i].b->cols,
				ConvLayers[cl].layer[i].b->channels);
			tmpVecW.push_back(tmpW);
			tmpVecb.push_back(tmpb);
		}
		cu_v_cvl_w.push_back(tmpVecW);
		cu_v_cvl_b.push_back(tmpVecb);
	}

	//���
	if(cuCorrect == NULL)
	{
		cuCorrect = new cuMatrix<int>(1,1,1);
		cuVote    = new cuMatrix<int>(testX.size(), Config::instance()->getSoftMax()[0]->m_numClasses, 1);
	}

	//���֮������
	cu_distortion_vector = new cuMatrixVector<double>();
	for(int i = 0; i < batch; i++){
		cu_distortion_vector->push_back(new cuMatrix<double>(ImgSize, ImgSize, Config::instance()->getChannels()));
	}
	cu_distortion_vector->toGpu();
}

void cuFreeCNNMemory(
	int batch,
	cuMatrixVector<double>&trainX, 
	cuMatrixVector<double>&testX,
	std::vector<cuCvl>&ConvLayers,
	std::vector<cuFll> &HiddenLayers, 
	cuSMR &smr)
{
	cuConvOutputSize.clear();
	cuPoolOutputSize.clear();
	cuKernelScan.clear();
	for(int i = 0; i < ConvLayers.size(); i++)
	{
		delete cuConv[i];
		delete cuPool[i];
		delete cuPointX[i];
		delete cuPointY[i];
	}
	cuConv.clear();
	cuPool.clear();
	cuPointX.clear();
	cuPointY.clear();

	delete cuPoolToFlActi;

	for(int i = 0; i < cuHiddenActi.size(); i++)
	{
		delete cuHiddenActi[i];
	}
	cuHiddenActi.clear();

	delete cuSoftMaxP;
	delete cuGroundTruth;
	delete cuSoftMaxDelta;

	for(int i = 0; i < HiddenLayers.size(); i++)
	{
		delete cuHiddenDelta[i];
	}
	cuHiddenDelta.clear();

	for(int i = 0; i < cuPoolDelta.size(); i++)
	{
		delete cuPoolDelta[i];
	}
	cuPoolDelta.clear();

	for(int i = 0; i < cuConvDelta.size(); i++)
	{
		delete cuConvDelta[i];
	}
	cuConvDelta.clear();
	
	for(int cl = 0; cl < cuPoolDeltaAndBorder.size(); cl++)
	{
		delete cuPoolDeltaAndBorder[cl];
	}
	cuPoolDeltaAndBorder.clear();

	for(int cl = 0; cl < cuConvLayerWgradTmp.size(); cl++)
	{
		delete cuConvLayerWgradTmp[cl];
	}
	cuConvLayerWgradTmp.clear();

	//����
	delete cu_v_smr_w;
	delete cu_v_smr_b;
	for(int i = 0; i < cu_v_hl_w.size(); i++)
	{
		delete cu_v_hl_w[i];
		delete cu_v_hl_b[i];
	}
	cu_v_hl_b.clear();
	cu_v_hl_w.clear();

	for(int cl = 0; cl < ConvLayers.size(); cl++)
	{
		for(int i = 0; i < ConvLayers[cl].layer.size(); i++)
		{
			delete cu_v_cvl_w[cl][i];
			delete cu_v_cvl_b[cl][i];
		}
		cu_v_cvl_w[cl].clear();
		cu_v_cvl_b[cl].clear();
	}
	cu_v_cvl_b.clear();
	cu_v_cvl_w.clear();
	//���

	//���֮������
	delete cu_distortion_vector;
}

/*
	����˵��   ����һ������ͳػ���ǰ�򴫵���GPU����
	arrayS     : ���������ݱ���Ϊ��ά����
	arrayW     : ��һ�����������Ȩֵ
	arrayB     : ��һ�������ƫ��ֵ
	conv       : �������
	pool       : �ػ������
	pointX	   : �ػ�֮����ѡ�е�x����¼���ڷ��򴫵���ʹ��
	pointY     : �ػ�֮����ѡ�е�y����¼���ڷ��򴫵���ʹ��
	inputSize  : ������ĳߴ�
	kernelSize : ���˵ĳߴ�
	convSize   : �������֮��ĳߴ�
	poolSize   : �ػ������֮��ĳߴ�
	poolingDim : �ػ������ÿ����СΪpoolingDim X PoolingDim��ͼƬѡȡһ�����ֵ��
	k1Amount   : ��һ������ľ��˸���Ĭ��Ϊ7��


	�����̷߳��䣺
	dim3(batch, cuKernelScan[0], Config::instance()->getChannels()), dim3(threadidx)
*/
__global__ void g_convAndPooling_1(
	double** arrayS,
	double** arrayW,
	double** arrayB,
	double* conv,
	double* pool,
	int* pointX,
	int* pointY,
	int inputSize,
	int kernelSize,
	int convSize,
	int poolSize,
	int poolingDim,
	int convArea,
	int poolArea,
	int batch,
	int k1Amount,
	int NONLIN)
{
	int sp = blockIdx.x;
	int k  = blockIdx.y;
	int c  = blockIdx.z;

	int convSize2  = convSize * convSize;
	int poolSize2  = poolSize * poolSize;
	int inputSize2 = inputSize* inputSize;
	int kernelSize2= kernelSize * kernelSize;

	int convSkip  = convArea * c + sp * k1Amount * convSize2 + k * convSize2;
	int poolSkip  = poolArea * c + sp * k1Amount * poolSize2 + k * poolSize2;

	double* curInput = arrayS[sp] + c * inputSize2;
	double* w        = arrayW[k]  + c * kernelSize2;
	double  b        = arrayB[k][c];

	double* curConv  = conv   + convSkip;
	double* curPool  = pool   + poolSkip;
	int* px          = pointX + poolSkip;
	int* py          = pointY + poolSkip;
	
	for(int tidx = 0; tidx < convSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < convSize2)
		{
			int x = idx / convSize;
			int y = idx % convSize;
			double val = 0.0;
			for(int i = 0; i < kernelSize; i++)
			{
				for(int j = 0; j < kernelSize; j++)
				{
					int xx = x + i;
					int yy = y + j;
					val += curInput[xx * inputSize + yy] * w[i * kernelSize + j];
				}
			}
			curConv[idx] = d_nonLinearity(val + b, NONLIN);
		}
	}
	
	__syncthreads();


	for(int tidx = 0; tidx < poolSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < poolSize2)
		{
			int x = idx / poolSize;
			int y = idx % poolSize;

			int curX = x * poolingDim;
			int curY = y * poolingDim;

			double _max = curConv[curX * convSize + curY];
			int lenx = min(convSize, (x + 1) * poolingDim);
			int leny = min(convSize, (y + 1) * poolingDim);

			for(int i = x * poolingDim; i < lenx; i++)
			{
				for(int j = y * poolingDim; j < leny; j++)
				{
					double val = curConv[i * convSize + j];
					if(_max < val){
						_max  = val;
						curX = i;
						curY = j;
					}
				}
			}
			px     [idx] = curX;
			py     [idx] = curY;
			curPool[idx] = _max;
		}
	}
}

/*
	����˵��   ���ڶ���������֮�󣩾���ͳػ���ǰ�򴫵���GPU����
	pool1      : ��һ������ػ�֮�����ݣ���Ϊ�ڶ������������
	arrayW     : �ڶ������������Ȩֵ
	arrayB     : �ڶ��������ƫ��ֵ
	conv2      : �ڶ����������
	pool2      : �ڶ����ػ������
	pointX	   : �ػ�֮����ѡ�е�x����¼���ڷ��򴫵���ʹ��
	pointY     : �ػ�֮����ѡ�е�y����¼���ڷ��򴫵���ʹ��
	pool1Size  : ������ĳߴ�
	kernelSize : ���˵ĳߴ�
	conv2Size  : �������֮��ĳߴ�
	pool2Size  : �ػ������֮��ĳߴ�
	poolingDim : �ػ������ÿ����СΪpoolingDim X PoolingDim��ͼƬѡȡһ�����ֵ��
	k1Amount   : ��һ������ľ��˸���Ĭ��Ϊ7��
	k2Amount   : �ڶ�������ľ��˸���Ĭ��Ϊ9��


	�����̷߳��䣺
	dim3(batch, cuKernelScan[i], Config::instance()->getChannels()),
	dim3(threadidx)
*/

__global__ void g_convAndPooling_2(
	double* pool1,
	double** arrayW,
	double** arrayB,
 	double* conv2,
 	double* pool2,
 	int* pointX,
 	int* pointY,
  	int pool1Size,
  	int kernelSize,
  	int conv2Size,
  	int pool2Size,
  	int poolingDim,
	int k1Scan,
	int k2Scan,
  	int k1Amount,
 	int k2Amount,
	int pool1Area,
	int conv2Area,
	int pool2Area,
	int NONLIN)
{
	int sp      = blockIdx.x;
	int c       = blockIdx.z;
	int k2 = blockIdx.y % k2Amount;
	int k1 = blockIdx.y / k2Amount;

	double* w   = arrayW[k2] + kernelSize * kernelSize * c;
	double  b   = arrayB[k2][c];

	int pool1Size2 = pool1Size * pool1Size;
	int conv2Size2 = conv2Size * conv2Size;
	int pool2Size2 = pool2Size * pool2Size;

	int pool2skip2 = pool2Area * c 
		+ sp * k2Scan * pool2Size2
		+ k1 * k2Amount * pool2Size2
		+ k2 * pool2Size2;

	double* pl1 = pool1
		+ pool1Area * c 
		+ sp * k1Scan * pool1Size2 
		+ k1 * pool1Size2;

	double* cv2 = conv2
		+ conv2Area * c
		+ sp * k2Scan * conv2Size2 
		+ k1 * k2Amount * conv2Size2
		+ k2 * conv2Size2;

	double* pl2 = pool2 + pool2skip2;
	int   * px  = pointX+ pool2skip2;
	int   * py  = pointY+ pool2skip2;

	for(int tidx = 0; tidx < conv2Size2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < conv2Size2)
		{
			int x = idx / conv2Size;
			int y = idx % conv2Size;
			double val = 0.0;
			for(int i = 0; i < kernelSize; i++)
			{
				for(int j = 0; j < kernelSize; j++)
				{
					int xx = x + i;
					int yy = y + j;
					cuAssert(xx <= pool1Size && yy <= pool1Size);
					val += pl1[xx * pool1Size + yy] * w[i * kernelSize + j];
				}
			}
			cv2[idx] = d_nonLinearity(val + b, NONLIN);
		}
	}
  	__syncthreads();

	for(int tidx = 0; tidx < pool2Size2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < pool2Size2)
		{
			int x = idx / pool2Size;
			int y = idx % pool2Size;

			int curX = x * poolingDim;
			int curY = y * poolingDim;

			double _max = cv2[curX * conv2Size + curY];
			int lenx = min(conv2Size, (x + 1) * poolingDim);
			int leny = min(conv2Size, (y + 1) * poolingDim);
			for(int i = x * poolingDim; i < lenx; i++)
			{
				for(int j = y * poolingDim; j < leny; j++)
				{
					double val = cv2[i * conv2Size + j];
					if(_max < val)
					{
						_max  = val;
						curX = i;
						curY = j;
					}
				}
			}
			px [idx] = curX;
			py [idx] = curY;
			pl2[idx] = _max;
		}
	}
}

void outputPoints(cuMatrix<int>* p)
{
	p->toCpu();
	for(int c = 0; c < p->channels; c++){
		for(int i = 0; i < p->rows; i++)
		{
			for(int j = 0; j < p->cols; j++)
			{
				printf("%d ", p->get(i,j, c));
			}printf("\n");
		}
		printf("%d\n");
	}
}

void outputMatrix(cuMatrix<double>* m)
{
	m->toCpu();
	for(int c = 0; c < m->channels; c++){
		for(int i = 0; i < m->rows; i++){
			for(int j = 0; j < m->cols; j++){
				printf("%.10lf ", m->get(i,j, c));
			}printf("\n");
		}
		printf("\n");
	}
}

void convAndPooling(double** x, std::vector<cuCvl> &CLayers, int batch, int ImgSize)
{
	//��һ��
	int curSize = ImgSize - Config::instance()->getConv()[0]->m_kernelSize + 1;//24
	int outputSize = curSize;//24

	int threadidx = min(outputSize * outputSize, 512);

	g_convAndPooling_1<<<
		dim3(batch, 
		cuKernelScan[0],
		Config::instance()->getChannels()),
		dim3(threadidx)>>>(
		x,
		CLayers[0].w->m_devPoint,
		CLayers[0].b->m_devPoint,
		cuConv[0]->devData,
		cuPool[0]->devData,
		cuPointX[0]->devData,
		cuPointY[0]->devData,
		ImgSize,
		Config::instance()->getConv()[0]->m_kernelSize,
		cuConvOutputSize[0],
		cuPoolOutputSize[0],
		Config::instance()->getConv()[0]->m_poolingDim,
		cuConv[0]->getArea(),
		cuPool[0]->getArea(),
		batch,
		Config::instance()->getConv()[0]->m_amount,
		Config::instance()->getNonLinearity()
	);

	cudaDeviceSynchronize();
	getLastCudaError("g_convAndPooling_1");

	for(int i = 1; i < Config::instance()->getConv().size(); i++)
	{
		int threadidx = min(cuConvOutputSize[i] * cuConvOutputSize[i], 512);

		g_convAndPooling_2<<<
			dim3(batch, cuKernelScan[i], Config::instance()->getChannels()),
			dim3(threadidx)>>>
			(cuPool[i - 1]->devData,
			CLayers[i].w->m_devPoint,
			CLayers[i].b->m_devPoint,
			cuConv[i]->devData,
			cuPool[i]->devData,
			cuPointX[i]->devData,
			cuPointY[i]->devData,
			cuPoolOutputSize[i - 1],
			Config::instance()->getConv()[i]->m_kernelSize,
			cuConvOutputSize[i],
			cuPoolOutputSize[i],
			Config::instance()->getConv()[i]->m_poolingDim,
			cuKernelScan[i - 1],
			cuKernelScan[i],
			Config::instance()->getConv()[i - 1]->m_amount,
			Config::instance()->getConv()[i]->m_amount,
			cuPool[i - 1]->getArea(),
			cuConv[i]->getArea(),
			cuPool[i]->getArea(),
			Config::instance()->getNonLinearity());

		cudaDeviceSynchronize();
		getLastCudaError("g_convAndPooling_2");
	}
}

/*
	�����ܣ����ؼ����
	acti    ������
	b       ��ƫ��ֵ

	�̷߳���<<<dim3(batch), dim3(NumHiddenNeurons)>>>
	batchΪ��ĸ���NumHiddenNeuronsΪ��ǰ������Ԫ�ĸ���
*/
__global__ void g_hiddenLayerActi(double* acti, double* b, int NONLIN)
{
	double* data  = acti + blockIdx.x * blockDim.x;
	double val = data[threadIdx.x];

	val = val + b[threadIdx.x];
	data[threadIdx.x] = d_nonLinearity(val, NONLIN);
}

__global__ void g_dropW(double * w, double * dropW, double* afterDropW, int len)
{
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			afterDropW[id] = dropW[id] * w[id];
		}
	}
}

/*
	Function name : cuMatrix(batch, size, channel) to cuMatrix(batch, size * channel, 1)
	cuPool        :
	cuPoolToFlActi:
	batch         :
	size          :
	channel       :
	threads       :<<<dim3(batch), dim3(256 = (size, channels))>>>
*/
__global__ void g_cuPoolToFlActi(double* cuPool, double*cuPoolToFlActi, int batch, int size, int channel)
{
	int b  = blockIdx.x;
	for(int i = 0; i < size * channel; i+=blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < size * channel)
		{
			int s = id / channel;
			int c = id % channel;
			//cuAssert((channel == 1 && c == 0) || channel > 1);
			cuPoolToFlActi[b * size * channel + size * c + s] = cuPool[c * batch * size + b * size + s];
		}
	}
}

void getHiddenLayerActi(std::vector<cuFll>&hLayers, cublasHandle_t handle)
{
	int poolidx = cuPool.size() - 1;
	g_cuPoolToFlActi<<<dim3(cuPool[poolidx]->rows), dim3(512)>>>
		(cuPool[poolidx]->devData, 
		cuPoolToFlActi->devData, 
		cuPool[poolidx]->rows,
		cuPool[poolidx]->cols,
		cuPool[poolidx]->channels);
	cudaDeviceSynchronize();

	getLastCudaError("g_cuPoolToFlActi");

	for(int hl = 0; hl < Config::instance()->getFC().size(); hl++)
	{
		g_dropW<<<1, 256>>>(hLayers[hl].W->devData,
			hLayers[hl].dropW->devData, 
			hLayers[hl].afterDropW->devData, 
			hLayers[hl].W->getLen());
		cudaDeviceSynchronize();
		getLastCudaError("g_dropW");

		if(hl == 0)
		{
			matrixMulTB(cuPoolToFlActi, 
				hLayers[hl].afterDropW, cuHiddenActi[hl], handle);
			if(cuHiddenActi[hl]->cols > MAX_THREADS){
				printf("g_hiddenLayerActi > MAX_THREADS\n");
				exit(0);
			}

			g_hiddenLayerActi<<<cuHiddenActi[hl]->rows, cuHiddenActi[hl]->cols>>>(cuHiddenActi[hl]->devData, 
				hLayers[hl].b->devData, Config::instance()->getNonLinearity());
			cudaDeviceSynchronize();
			getLastCudaError("g_hiddenLayerActi");
		}
		else 
		{
			matrixMulTB(cuHiddenActi[hl - 1], 
				hLayers[hl].afterDropW, cuHiddenActi[hl], handle);

			if(cuHiddenActi[hl]->cols > MAX_THREADS){
				printf("g_hiddenLayerActi > MAX_THREADS\n");
				exit(0);
			}

			g_hiddenLayerActi<<<cuHiddenActi[hl]->rows, cuHiddenActi[hl]->cols>>>(cuHiddenActi[hl]->devData, 
				hLayers[hl].b->devData, Config::instance()->getNonLinearity());
			cudaDeviceSynchronize();
			getLastCudaError("g_hiddenLayerActi");
		}
	}
}

/*
	�����ܣ�����ع������
	softMaxp: �ع����ݴ�Ŵ�
	b       : �ع���ƫ����
	�̷߳��䣺<<<cuSoftMaxP->rows, cuSoftMaxP->cols>>>
*/
__global__ void g_getSoftMaxP(double* softMaxP, double* b)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	extern __shared__ double _share[];

	double * _max = _share;
	double * _sum = _share + blockDim.x;

	double* sp = softMaxP + bid * blockDim.x;
	sp[tid] += b[tid];
	_max[tid] = sp[tid];

	int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(tid < (len >> 1))
		{
			if(_max[tid] < _max[tid + skip])
			{
				_max[tid] = _max[tid + skip];
			}
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();

	sp[tid] -= _max[0];
	sp[tid] = exp(sp[tid]);
	_sum[tid] = sp[tid];

	len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(tid < (len >> 1))
		{
			_sum[tid] += _sum[tid + skip];
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();
	sp[tid] /= _sum[0];
}

void getSoftMaxP(cuSMR& smr, cublasHandle_t handle)
{
	matrixMulTB(cuHiddenActi[Config::instance()->getFC().size() - 1],
		smr.Weight, cuSoftMaxP, handle);

	if(cuSoftMaxP->cols > MAX_THREADS){
		printf("g_getSoftMaxP > MAX_THREADS\n");
		exit(0);
	}

	g_getSoftMaxP<<<cuSoftMaxP->rows, cuSoftMaxP->cols, sizeof(double) * cuSoftMaxP->cols * 2>>>(cuSoftMaxP->devData, smr.b->devData);
	cudaDeviceSynchronize();
	getLastCudaError("g_getSoftMaxP");
}

/*
	������   ������ع��Ĵ��
	softMaxP   : �ع�㼤��ֵ
	groundTruth: ��ǩ�ľ���
	cost       : ��۽��д������
	y          : b��ǩ
	rows       : ��ݵ��У�
	cols       : ��ݵ���
	batch      : ��ĸ���
	�̷߳���   �������漰����Ͳ�������������������ر�����Կ���
				<<<dim3(1),dim3(256)>>>������Ա���Ҫȥͬ����ͬblock֮������
*/
__global__ void g_getCost_1(double* softMaxP,
	double* groundTruth, double* cost, int*y, int rows, int cols, int batch)
{
	extern __shared__ double _sum[];
	//grounTruth ����
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
	//grounTruth��ֵ
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

	//���
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

/*
	�����ܣ�softMax���Ȩ�ش��
	cost    : ��۵����
	weight  : Ȩ��
	lambda  : ϵ��
	rows    : Ȩ�ص�����
	cols    : Ȩ�ص�����
	�̷߳��䣺<<<dim3(1),dim3(256)>>>
*/
 __global__ void g_getCost_2(double* cost,
 	double* weight,
 	double lambda, int rows, int cols)
 {
 	extern __shared__ double _sum[];
 	_sum[threadIdx.x] = 0;
 	__syncthreads();
 
	int len = rows * cols;
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

 __inline__ __device__ double atomicAdd(double* address, double val)
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

 /*
	�����ܣ�����Ĵ�ۺ���ļ���
	cost    : ��۵����
	weight  : Ȩ��
	lambda  : ϵ��
	rows    : Ȩ�ص�����
	cols    : Ȩ�ص�����
	�̷߳��䣺<<<dim3(���˵ĸ���),dim3(256)>>>
*/
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

void getCost(
	int*y,
	std::vector<cuCvl> &CLayers, 
	std::vector<cuFll> &hLayers,
	cuSMR &smr,
	double lambda,
	int batch)
{
	g_getCost_1<<<dim3(1), dim3(256), sizeof(double) * 256>>>(cuSoftMaxP->devData, cuGroundTruth->devData,
		smr.cost->devData, y, cuSoftMaxP->rows, cuSoftMaxP->cols, batch);
	cudaDeviceSynchronize();

	g_getCost_2<<<dim3(1), dim3(256), sizeof(double) * 256>>>(smr.cost->devData,  smr.Weight->devData, lambda,
		smr.Weight->rows, smr.Weight->cols);
	cudaDeviceSynchronize();

	for(int cl = 0; cl < CLayers.size(); cl++)
	{
		g_getCost_3<<<dim3(Config::instance()->getConv()[cl]->m_amount), dim3(32), sizeof(double) * 32>>>(smr.cost->devData, CLayers[cl].w->m_devPoint, lambda,
			Config::instance()->getConv()[cl]->m_kernelSize, Config::instance()->getConv()[cl]->m_kernelSize);
		cudaDeviceSynchronize();
	}
	getLastCudaError("g_getCost_3");
}


__global__ void g_getSoftMaxDelta(double* softMaxDelta, double* softMaxP, double* groudTruth, int len)
{
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			softMaxDelta[id] = softMaxP[id] - groudTruth[id];
		}
	}
}

/*
	�̰߳���:<<<dim3(1),dim3(256)>>>
*/

__global__ void g_getSmrWgrad(double* wgrad, double* weight, double lambda, int len, int batch)
{
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			double val = wgrad[id] / batch;
			val += lambda * weight[id];
			wgrad[id] = val;
		}
	}
}

/*
	�����ܣ����bgrad
	�̷߳���<<<dim3(������),dim3(����)>>>
	ÿ��block����һ������,���ö��ֵķ���
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
	�����ܣ����bgrad
	�̷߳���<<<dim3(������),dim3(����)>>>
	ÿ��block����һ������,���ö��ֵķ���
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

void getSoftMaxDelta(cuSMR &smr, double lambda, int batch, cublasHandle_t handle)
{
	g_getSoftMaxDelta<<<dim3(1), dim3(256)>>>(cuSoftMaxDelta->devData,
		cuSoftMaxP->devData,
		cuGroundTruth->devData, cuSoftMaxDelta->getLen());
	cudaDeviceSynchronize();

	matrixMulTA(cuSoftMaxDelta, cuHiddenActi[Config::instance()->getFC().size() - 1], smr.Wgrad, handle);

	g_getSmrWgrad<<<dim3(1), dim3(256)>>>(smr.Wgrad->devData,
		smr.Weight->devData, lambda, smr.Wgrad->getLen(), batch);
	cudaDeviceSynchronize();

	if(cuSoftMaxDelta->rows > MAX_THREADS)
	{
		printf("getSoftMaxDelta g_getBgrad > MAX_THREADS\n");
		exit(0);
	}
 	g_getBgrad<<<dim3(cuSoftMaxDelta->cols), dim3(cuSoftMaxDelta->rows), 
		sizeof(double) * cuSoftMaxDelta->rows>>>(
		cuSoftMaxDelta->devData, 
 		smr.bgrad->devData,
		batch);
 	cudaDeviceSynchronize();
	getLastCudaError("getSoftMaxDelta");
}

/*
	threads: <<<dim3(256),dim3(256)>>>
*/
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

__global__ void g_getHiddlenWgrad(double* wgrad, double* dropM, int len, int batch)
{
	for(int i = 0; i < len; i += blockDim.x * gridDim.x)
	{
		int id = i + blockDim.x * blockIdx.x + threadIdx.x;
		if(id < len)
		{
			wgrad[id] /= batch;
			wgrad[id] *= dropM[id];
		}
	}
}

void getHiddenDelta(
	std::vector<cuFll> &hLayers,
	cuSMR &smr,
	double lambda,
	int batch,
	cublasHandle_t handle)
{
	for(int hl = Config::instance()->getFC().size() - 1; hl >= 0; hl--)
	{
		if(hl == Config::instance()->getFC().size() - 1)
		{
 			matrixMul(cuSoftMaxDelta,
 				smr.Weight, cuHiddenDelta[hl], handle);
 			g_dnonLinearity<<<dim3(256), dim3(256)>>>(cuHiddenDelta[hl]->devData,
				cuHiddenActi[hl]->devData, cuHiddenDelta[hl]->getLen(), Config::instance()->getNonLinearity());
 			cudaDeviceSynchronize();
		}
		else
		{
			matrixMul(cuHiddenDelta[hl + 1], hLayers[hl + 1].afterDropW,
				cuHiddenDelta[hl], handle);
			g_dnonLinearity<<<dim3(256), dim3(256)>>>(
				cuHiddenDelta[hl]->devData, 
				cuHiddenActi[hl]->devData,
				cuHiddenDelta[hl]->getLen(), Config::instance()->getNonLinearity());
			cudaDeviceSynchronize();
			getLastCudaError("g_dnonLinearity");
		}
	}

	for(int hl = Config::instance()->getFC().size() - 1; hl >= 0; hl--)
	{
		if(hl != 0)
		{
			matrixMulTA(cuHiddenDelta[hl],
				cuHiddenActi[hl - 1],
				hLayers[hl].Wgrad, handle);
			g_getHiddlenWgrad<<<dim3(256), dim3(256)>>>(hLayers[hl].Wgrad->devData, hLayers[hl].dropW->devData,
				hLayers[hl].Wgrad->getLen(), batch);
			cudaDeviceSynchronize();
			getLastCudaError("g_getHiddlenWgrad");
		}
		else
		{
			matrixMulTA(cuHiddenDelta[hl],
				cuPoolToFlActi,
				hLayers[hl].Wgrad, handle);
			g_getHiddlenWgrad<<<dim3(256), dim3(256)>>>(hLayers[hl].Wgrad->devData, hLayers[hl].dropW->devData, 
				hLayers[hl].Wgrad->getLen(), batch);
			cudaDeviceSynchronize();
			getLastCudaError("g_getHiddlenWgrad");
		}

		if(cuHiddenDelta[hl]->rows > MAX_THREADS)
		{
			printf("getHiddenDelta g_getBgrad > MAX_THREADS\n");
			exit(0);
		}
		g_getBgrad<<<dim3(cuHiddenDelta[hl]->cols), dim3(cuHiddenDelta[hl]->rows),
			sizeof(double) * cuSoftMaxDelta->rows>>>
			(cuHiddenDelta[hl]->devData, hLayers[hl].bgrad->devData, batch);
		cudaDeviceSynchronize();
		getLastCudaError("g_getBgrad");
	}

	
}

/*
	function�����򴫵��е�unpooling��������Ҫ���
			  ��¼��pointx��pointy��ݣ���delta��ػ����
			  ��һ�㴫��
	threads��<<<dim3(min(512), dim3(min(512))>>>
*/

__global__ void g_unPooling(int* pointX, int* pointY,
	double* _pool, double* _conv,
	int poolSize, int poolDim, int convSize, int len)
{
	int poolSize2 = poolSize * poolSize;
	int convSize2 = convSize * convSize;

	for(int i = 0; i < len; i += gridDim.x * blockDim.x)
	{
		int id = i + blockDim.x * blockIdx.x + threadIdx.x;
		if(id < len)
		{
			int convId = id / poolSize2;
			int idx    = id % poolSize2;
			int poolSkip = poolSize2 * convId;
			int*       x = pointX  + poolSkip;
			int*       y = pointY  + poolSkip;
			double* pool = _pool   + poolSkip;
			double* conv = _conv   + convSize2 * convId;
			int    curX = x   [idx];
			int    curY = y   [idx];
			double curP = pool[idx];
			cuAssert(curX < convSize && curY < convSize);

			conv[curX * convSize + curY] = curP;
		}
	}
}

/*
	function�����ڶ����convdelta�����Ƶ�����һ���pooldelta��
			  ������Ҫ����ڶ���ĺ˺�����Ҫ���˺�����ת180�ȡ�
	threads ��<<<dim3(batch, k1 * k2, channel),dim3(nxtSize, nxtSize)>>>
			  �����漰�ļӷ�(reduce������,����ÿ���߳�ֻ�����ۼ�k1��Ԫ�أ�
			  ����ֱ����ԭ�Ӳ����Ϳ��ԴﵽĿ�ģ��ۼ�Ԫ�ع�࣬
			  ����Ҫ���ö��ֵķ�����ʵ�֣�

*/
__global__ void g_dConvAdd(
	double* _convDelta,
	double* _addBorder,
	double**_w,
	double* _poolDelta,
	int     _convOutputSize,
	int     _poolOutputSize,
	int     _kernelScan1,
	int     _kernelScan2,
	int     _kernelAmount1,
	int     _kernelAmount2,
	int     _kernelSize,
	int     _convDeltaArea,
	int     _addBorderArea,
	int     _poolDeltaArea)  
{
	int curSize          = _convOutputSize;
	int curAddBorderSize = _poolOutputSize;
	int wSize            = _kernelSize;
	int nxtSize          = _poolOutputSize;
	int k1 = blockIdx.y / _kernelAmount2;
	int k2 = blockIdx.y % _kernelAmount2;
	int s  = blockIdx.x;
	int c  = blockIdx.z;
	int curSize2 = curSize * curSize;
	int nxtSize2 = nxtSize * nxtSize;

	double* curDelta = _convDelta 
		+ c * _convDeltaArea
		+ curSize2 * s * _kernelScan2
		+ curSize2 * k1* _kernelAmount2
		+ curSize2 * k2;
	double* nxtDelta = _poolDelta 
		+ c * _poolDeltaArea
		+ nxtSize2 * s * _kernelScan1
		+ nxtSize2 * k1;
	double* addBorder= _addBorder	
		+ c * _addBorderArea
		+ nxtSize2 * s * _kernelScan2
		+ nxtSize2 * k1* _kernelAmount2
		+ nxtSize2 * k2;
	double*        w = _w[k2] + c * _kernelSize * _kernelSize;

	for(int tidx = 0; tidx < curSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < curSize2)
		{
			int i = idx / curSize;
			int j = idx % curSize;
			double val = curDelta[idx];
			int x = i + (wSize >> 1);
			int y = j + (wSize >> 1);
			addBorder[x * curAddBorderSize + y] = val;
		}
	}
	__syncthreads();

	for(int tidx = 0; tidx < nxtSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < nxtSize2)
		{
			int i = idx / nxtSize;
			int j = idx % nxtSize;
			double val = 0.0;
			for(int x = 0; x < wSize; x++)
			{
				for(int y = 0; y < wSize; y++)
				{
					int cx = i + x - (wSize >> 1);
					int cy = j + y - (wSize >> 1);
					int wx = wSize - x - 1;
					int wy = wSize - y - 1;
					if(cx >= 0 && cx < curAddBorderSize && cy >= 0 && cy < curAddBorderSize)
					{
						val += addBorder[cx * curAddBorderSize + cy] * w[wx * wSize + wy];
					}
				}
			}
			atomicAdd(nxtDelta + idx, val);
		}
	}
}

/*
	
	�����ܣ��������wgrad�ĵ�һ�����������漰�������Ƚϴ��reduce���������Բ�����ֱ����actom������
			  ����
	�̷߳��䣺dim3<<<dim3(batch, kernelAmount2, channels), dim3(nxtSize * nxtSize, kernelAmount1)>>>
*/
__global__ void g_conv(double* pool,
	double* convDelta,
	double* WgradTmp,
	int poolOutputSize,
	int convOutputSize,
	int kernelScan1,
	int kernelScan2,
	int kernelAmount1,
	int kernelAmount2,
	int kernelSize,
	int poolArea,
	int convDeltaArea,
	int wgradTmpArea)
{
	int c = blockIdx.z;
	int s = blockIdx.x;
	int k2= blockIdx.y % kernelAmount2;
	int k1= blockIdx.y / kernelAmount2;


	int curSize = poolOutputSize;
	int wSize   = convOutputSize;
	int nxtSize = kernelSize;

	int curSize2 = curSize * curSize;
	int wSize2   = wSize   * wSize;
	int nxtSize2 = nxtSize * nxtSize;

	double* cur   = pool
		+ c * poolArea
		+ curSize2 * s * kernelScan1
		+ curSize2 * k1;

	double* w     = convDelta
		+ c * convDeltaArea
		+ wSize2 * s * kernelScan2
		+ wSize2 * k1* kernelAmount2
		+ wSize2 * k2;

	double* nxt   = WgradTmp
		+ c * wgradTmpArea
		+ nxtSize2 * s * kernelScan2
		+ nxtSize2 * k1* kernelAmount2
		+ nxtSize2 * k2;

	for(int tidx = 0; tidx < nxtSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < nxtSize2)
		{
			int i = idx / nxtSize;
			int j = idx % nxtSize;

			double val = 0.0;
			for(int x = 0; x < wSize; x++)
			{
				for(int y = 0; y < wSize; y++)
				{
					int cx = i + x;
					int cy = j + y;
					val += cur[cx * curSize + cy] * w[x * wSize + y];
				}
			}
			nxt[idx] = val;
		}
	}
}

/*
	�̷߳��䣺<<<dim3(k2, kernelSize*kernelSize, channels), dim3(256)>>>
*/
__global__ void g_convAdd(
	double* WgradTmp, 
	double** Wgrad,
	double** w,
	int kernelScan1,
	int kernelScan2,
	int kernelAmount1,
	int kernelAmount2,
	int kernelSize,
	int batch,
	int wgradTmpArea,
	int wgradArea,
	int wArea,
	double lambda)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	int kid= blockIdx.y;
	int c  = blockIdx.z;

	_sum[threadIdx.x] = 0;
	__syncthreads();
	int kernelSize2 = kernelSize * kernelSize;
	int  tlen = batch * kernelScan1;
	for(int i = 0; i <  tlen; i += blockDim.x)
	{
		int idx = i + threadIdx.x;
		if(idx < tlen)
		{
			int s = idx / kernelScan1;
			int k1= idx % kernelScan1;

			int id = 
				c * wgradTmpArea
				+ kernelSize2 * s * kernelScan2
				+ kernelSize2 * k1* kernelAmount2
				+ kernelSize2 * k2 + kid;
			_sum[threadIdx.x] += WgradTmp[id];
		}
	}
	__syncthreads();

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
	__syncthreads();

	if(threadIdx.x == 0)
	{
		Wgrad[k2][kid + c * wgradArea] = _sum[0] / batch + w[k2][kid + c * wArea] * lambda;
	}
}

/*
	�̷߳��䣺<<<dim3(kernelAmount2, channels), dim3(256)>>>
*/
__global__ void g_getCLayerBgrad(double* delta,
	double** bgrad,
	int deltaSize,
	int kernelScan1,
	int kernelScan2,
	int kernelAmount1,
	int kernelAmount2,
	int batch,
	int deltaArea)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	int c  = blockIdx.y;

	_sum[threadIdx.x] = 0.0;
	__syncthreads();

	int deltaSize2 = deltaSize * deltaSize;
	int tlen = batch * kernelScan1 * deltaSize2;

	for(int i = 0; i < tlen; i += blockDim.x)
	{
		int idx = i + threadIdx.x;
		if(idx < tlen)
		{
			int t1 = idx / deltaSize2;//s,kernel1
			int t2 = idx % deltaSize2;//x,y
			int s  = t1 / kernelScan1;
			int k1 = t1 % kernelScan1;
			int id = 
				c * deltaArea
				+ deltaSize2 * s * kernelScan2
				+ deltaSize2 * k1* kernelAmount2
				+ deltaSize2 * k2
				+ t2;
			
			_sum[threadIdx.x] += delta[id];
		}
	}
	__syncthreads();

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
	__syncthreads();


	if(threadIdx.x == 0)
	{
		bgrad[k2][c] = _sum[0] / batch;
	}
}

/*
	
	�����ܣ��������wgrad�ĵ�һ�����������漰�������Ƚϴ��reduce������
			���Բ�����ֱ����actom���������ǲ��ö��������
	<<<dim3(batch, cuKernelScan[cl], Config::instance()->getChannels()),
			dim3(threadidx)>>>
*/
__global__ void g_conv_1(double** sArray,
	double* convDelta,
	double* WgradTmp,
	int imgSize,
	int convOutputSize,
	int kernelScan2,
	int kernelAmount1,
	int kernelSize,
	int sArrayArea,
	int convDeltaArea,
	int wgrapTmpArea)
{
	int curSize = imgSize;
	int wSize   = convOutputSize;
	int nxtSize = kernelSize;

	int s = blockIdx.x;
	int k2= blockIdx.y;
	int c = blockIdx.z;

	int wSize2   = wSize * wSize;
	int nxtSize2 = nxtSize * nxtSize;
	double* cur  = sArray[s] + c * sArrayArea;

	double* w     = convDelta
		+ c * convDeltaArea
		+ wSize2 * s * kernelScan2
		+ wSize2 * k2;

	double* nxt   = WgradTmp
		+ c * wgrapTmpArea
		+ nxtSize2 * s * kernelScan2
		+ nxtSize2 * k2;

	for(int tidx = 0; tidx < nxtSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < nxtSize2)
		{
			int i = idx / nxtSize;
			int j = idx % nxtSize;
			double val = 0.0;
			for(int x = 0; x < wSize; x++)
			{
				for(int y = 0; y < wSize; y++)
				{
					int cx = i + x;
					int cy = j + y;
					val += cur[cx * curSize + cy] * w[x * wSize + y];
				}
			}
			nxt[idx] = val;
		}
	}
}


/*
	�̷߳��䣺<<<dim3(k1, kernelSize*kernelSize, channels), dim3(256)>>>
*/
__global__ void g_convAdd_1(double* WgradTmp, double** Wgrad,
	double** w,
	int kernelScan2,
	int kernelAmount2,
	int kernelSize,
	int batch,
	double lambda,
	int wgradTmpArea,
	int wgradArea,
	int wArea)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	int kid= blockIdx.y;
	int c  = blockIdx.z;

	int tid= threadIdx.x;

	_sum[threadIdx.x] = 0;
	__syncthreads();
	int kernelSize2 = kernelSize * kernelSize;
	int tlen = batch;
	for(int i = 0; i < tlen; i += blockDim.x)
	{
		int s = i + threadIdx.x;
		if(s < tlen)
		{
			int id = 
				c * wgradTmpArea
				+ kernelSize2 * s * kernelScan2
				+ kernelSize2 * k2 + kid;
			_sum[threadIdx.x] += WgradTmp[id];
		}
	}
	__syncthreads();

	int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(tid < (len >> 1))
		{
			_sum[tid] += _sum[tid + skip];
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();

	if(tid == 0)
	{
		Wgrad[k2][kid + c * wgradArea] = _sum[0] / batch + w[k2][kid + c * wArea] * lambda;
	}
}

/*
	�̷߳��䣺<<<dim3(kernelAmount2, channels), dim3(256)>>>
*/
__global__ void g_getCLayerBgrad_1(double* delta,
	double** bgrad,
	int deltaSize,
	int kernelScan2,
	int kernelAmount2,
	int batch,
	int deltaArea,
	int bgradArea)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	int c  = blockIdx.y;
	_sum[threadIdx.x] = 0.0;
	__syncthreads();

	int deltaSize2 = deltaSize * deltaSize;
	int tlen = deltaSize2 * batch;
	for(int i = 0; i < tlen; i += blockDim.x)
	{
		int idx = i + threadIdx.x;
		if(idx < tlen)
		{
			int s  = idx / (deltaSize2);//s
			int t2 = idx % (deltaSize2);//x,y

			int id = 
				deltaArea * c
				+ deltaSize2 * s * kernelScan2
				+ deltaSize2 * k2
				+ t2;
			_sum[threadIdx.x] += delta[id];
		}
	}
	__syncthreads();

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
	__syncthreads();


	if(threadIdx.x == 0)
	{
		bgrad[k2][c] = _sum[0] / batch;
	}
}

/*
	function: g_cuPoolFlDelta
	threads : <<<dim3(batch), dim3(512)>>> 
*/
__global__ void g_cuPoolFlDelta(double* cuPoolFlDelta, 
	double* cuPoolDelta, int batch, int size, int channels)
{
	int b = blockIdx.x;
	for(int i = 0; i < size * channels; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < size * channels)
		{
			int s = id / channels;
			int c = id % channels;
			cuPoolDelta[c * batch * size + b * size + s] = cuPoolFlDelta[b * size * channels + size * c + s];
		}
	}
}

void dConvAndUnpooling(double**x, 
	std::vector<cuCvl> &CLayers,
	std::vector<cuFll> &hLayers,
	double lambda,
	int batch, int ImgSize, int nclasses, cublasHandle_t handle)
{
	matrixMul(cuHiddenDelta[0], hLayers[0].afterDropW, 
		cuPoolToFlDelta, handle);

	int threads = min(512, cuPoolDelta[cuPoolDelta.size() - 1]->channels * cuPoolDelta[cuPoolDelta.size() - 1]->cols);

	g_cuPoolFlDelta<<<dim3(cuPoolToFlDelta->rows), dim3(threads)>>>(
		cuPoolToFlDelta->devData, 
		cuPoolDelta[cuPoolDelta.size() - 1]->devData,
		cuPoolDelta[cuPoolDelta.size() - 1]->rows,
		cuPoolDelta[cuPoolDelta.size() - 1]->cols,
		cuPoolDelta[cuPoolDelta.size() - 1]->channels);
	cudaDeviceSynchronize();
	getLastCudaError("g_cuPoolFlDelta");

	for(int cl = CLayers.size() - 1; cl >= 0; cl--)
	{
		cuConvDelta[cl]->gpuClear();
		int len = cuPoolDelta[cl]->getLen();
		g_unPooling<<<dim3(std::min((len + 511) / 512, 512)),
			dim3(512)>>>(
			cuPointX[cl]->devData,
			cuPointY[cl]->devData,
			cuPoolDelta[cl]->devData,
			cuConvDelta[cl]->devData,
			cuPoolOutputSize[cl],
			Config::instance()->getConv()[cl]->m_poolingDim,
			cuConvOutputSize[cl],
			len);
	
		cudaDeviceSynchronize();
		getLastCudaError("g_unPooling");

		g_dnonLinearity<<<dim3(256),dim3(256)>>>(cuConvDelta[cl]->devData,
			cuConv[cl]->devData, cuConvDelta[cl]->getLen(), Config::instance()->getNonLinearity());
		cudaDeviceSynchronize();
		getLastCudaError("g_dnonLinearity");

		if(cl > 0)
		{
			cuPoolDelta[cl - 1]->gpuClear();
			cuPoolDeltaAndBorder[cl - 1]->gpuClear();
			int threadidx = min(cuPoolOutputSize[cl - 1] * cuPoolOutputSize[cl - 1], 512);
			
			g_dConvAdd<<<dim3(batch, cuKernelScan[cl], Config::instance()->getChannels()),
				dim3(threadidx)>>>
			(
				cuConvDelta[cl]->devData,
				cuPoolDeltaAndBorder[cl - 1]->devData,
				CLayers[cl].w->m_devPoint,
				cuPoolDelta[cl - 1]->devData,
				cuConvOutputSize[cl],
				cuPoolOutputSize[cl - 1],
				cuKernelScan[cl - 1],
				cuKernelScan[cl],
				Config::instance()->getConv()[cl - 1]->m_amount,
				Config::instance()->getConv()[cl]->m_amount,
				Config::instance()->getConv()[cl]->m_kernelSize,
				cuConvDelta[cl]->getArea(),
				cuPoolDeltaAndBorder[cl - 1]->getArea(),
				cuPoolDelta[cl - 1]->getArea());
			cudaDeviceSynchronize();
			getLastCudaError("g_dConvAdd");
			
			threadidx = min(Config::instance()->getConv()[cl]->m_kernelSize * Config::instance()->getConv()[cl]->m_kernelSize, 512);

			g_conv<<<dim3(batch, cuKernelScan[cl], Config::instance()->getChannels()),
				dim3(threadidx)>>>(
				cuPool[cl - 1]->devData,
				cuConvDelta[cl]->devData,
				cuConvLayerWgradTmp[cl]->devData,
				cuPoolOutputSize[cl - 1],
				cuConvOutputSize[cl],
				cuKernelScan[cl - 1],
				cuKernelScan[cl],
				Config::instance()->getConv()[cl - 1]->m_amount,
				Config::instance()->getConv()[cl]->m_amount,
				Config::instance()->getConv()[cl]->m_kernelSize,
				cuPool[cl - 1]->getArea(),
				cuConvDelta[cl]->getArea(),
				cuConvLayerWgradTmp[cl]->getArea());
			cudaDeviceSynchronize();
			getLastCudaError("g_conv");

			g_convAdd<<<dim3(Config::instance()->getConv()[cl]->m_amount,
				Config::instance()->getConv()[cl]->m_kernelSize * Config::instance()->getConv()[cl]->m_kernelSize,
				Config::instance()->getChannels()),
				dim3(256),
				sizeof(double) * 256>>>(
				cuConvLayerWgradTmp[cl]->devData,
				CLayers[cl].wgrad->m_devPoint,
				CLayers[cl].w->m_devPoint,
				cuKernelScan[cl - 1],
				cuKernelScan[cl],
				Config::instance()->getConv()[cl - 1]->m_amount,
				Config::instance()->getConv()[cl]->m_amount,
				Config::instance()->getConv()[cl]->m_kernelSize,
				batch,
				cuConvLayerWgradTmp[cl]->getArea(),
				CLayers[cl].layer[0].Wgrad->getArea(),
				CLayers[cl].layer[0].W->getArea(),
				lambda);
			cudaDeviceSynchronize();
			getLastCudaError("g_convAdd");

			g_getCLayerBgrad<<<dim3(Config::instance()->getConv()[cl]->m_amount,
				Config::instance()->getChannels()), 
				dim3(256),
				sizeof(double) * 256>>>(cuConvDelta[cl]->devData,
				CLayers[cl].bgrad->m_devPoint,
				cuConvOutputSize[cl],
				cuKernelScan[cl - 1],
				cuKernelScan[cl],
				Config::instance()->getConv()[cl - 1]->m_amount,
				Config::instance()->getConv()[cl]->m_amount,
				batch,
				cuConvDelta[cl]->getArea());
			cudaDeviceSynchronize();
			getLastCudaError("g_getCLayerBgrad");
		}
		else
		{
			int threadidx = min(Config::instance()->getConv()[cl]->m_kernelSize 
				* Config::instance()->getConv()[cl]->m_kernelSize, 
				512);
			g_conv_1<<<dim3(batch, cuKernelScan[cl], Config::instance()->getChannels()),
				dim3(threadidx)>>>
				(x,
				cuConvDelta[cl]->devData,
				cuConvLayerWgradTmp[cl]->devData,
				ImgSize,
				cuConvOutputSize[cl],
				cuKernelScan[cl],
				Config::instance()->getConv()[cl]->m_amount,
				Config::instance()->getConv()[cl]->m_kernelSize,
				ImgSize * ImgSize,
				cuConvDelta[cl]->getArea(),
				cuConvLayerWgradTmp[cl]->getArea());
			cudaDeviceSynchronize();
			getLastCudaError("g_conv_1");

			g_convAdd_1<<<dim3(Config::instance()->getConv()[cl]->m_amount, 
				Config::instance()->getConv()[cl]->m_kernelSize * Config::instance()->getConv()[cl]->m_kernelSize,
				Config::instance()->getChannels()),
				dim3(256),
				sizeof(double) * 256>>>(
				cuConvLayerWgradTmp[cl]->devData,
				CLayers[cl].wgrad->m_devPoint,
				CLayers[cl].w->m_devPoint,
				cuKernelScan[cl],
				Config::instance()->getConv()[cl]->m_amount,
				Config::instance()->getConv()[cl]->m_kernelSize,
				batch,
				lambda,
				cuConvLayerWgradTmp[cl]->getArea(),
				CLayers[cl].layer[0].Wgrad->getArea(),
				CLayers[cl].layer[0].W->getArea());
			cudaDeviceSynchronize();
			getLastCudaError("g_convAdd_1");


			g_getCLayerBgrad_1<<<dim3(Config::instance()->getConv()[cl]->m_amount, 
				Config::instance()->getChannels()),
				dim3(256), 
				sizeof(double) * 256>>>
				(cuConvDelta[cl]->devData,
				CLayers[cl].bgrad->m_devPoint, 
				cuConvOutputSize[cl], 
				cuKernelScan[cl],
				Config::instance()->getConv()[cl]->m_amount, 
				batch, 
				cuConvDelta[cl]->getArea(),
				CLayers[cl].layer[0].bgrad->getArea());
			cudaDeviceSynchronize();
			getLastCudaError("g_getCLayerBgrad_1");
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

void updataWB(std::vector<cuCvl> &CLayers, 
	std::vector<cuFll> &hLayers,
	cuSMR &smr,
	double lrate,
	double momentum,
	int batch)
{
	g_vecAdd<<<dim3(min((cu_v_smr_w->getLen() + 255) / 256, 5120)),
		dim3(256)>>>(
		cu_v_smr_w->devData, 
		smr.Wgrad->devData, 
		smr.Weight->devData,
		cu_v_smr_b->devData, 
		smr.bgrad->devData, 
		smr.b->devData, 
		smr.Wgrad->getLen(),
		smr.bgrad->getLen(),
		momentum, 
		lrate);
	for(int i = 0; i < hLayers.size(); i++)
	{
		g_vecAdd<<<dim3(min((cu_v_hl_w[i]->getLen() + 255) / 256, 5120)),dim3(256)>>>(cu_v_hl_w[i]->devData, hLayers[i].Wgrad->devData, hLayers[i].W->devData,
			cu_v_hl_b[i]->devData, hLayers[i].bgrad->devData, hLayers[i].b->devData,
			hLayers[i].Wgrad->getLen(), hLayers[i].bgrad->getLen(), momentum, lrate);
	}
 	for(int cl = 0; cl < CLayers.size(); cl++)
 	{
 		for(int i = 0; i < CLayers[cl].layer.size(); i++)
 		{
 			g_vecAdd<<<dim3(1),dim3(64)>>>(cu_v_cvl_w[cl][i]->devData, CLayers[cl].layer[i].Wgrad->devData, CLayers[cl].layer[i].W->devData,
 				cu_v_cvl_b[cl][i]->devData, CLayers[cl].layer[i].bgrad->devData, CLayers[cl].layer[i].b->devData,
 				cu_v_cvl_w[cl][i]->getLen(), cu_v_cvl_b[cl][i]->getLen(), momentum, lrate);
 		}
 	}
 	cudaDeviceSynchronize();
	getLastCudaError("updateWB");
}

void getNetworkCost(double** x, 
	int* y , 
	std::vector<cuCvl> &CLayers, 
	std::vector<cuFll> &hLayers,
	cuSMR &smr,
	double lambda,
	int batch,
	int ImgSize, 
	int nclasses,
	cublasHandle_t handle)
{
	convAndPooling(x, CLayers, batch, ImgSize);
	getHiddenLayerActi(hLayers, handle);
	getSoftMaxP(smr, handle);
	getCost(y,CLayers,hLayers,smr,lambda,batch);
	getSoftMaxDelta(smr,lambda, batch, handle);
	getHiddenDelta(hLayers, smr, lambda, batch, handle);
	dConvAndUnpooling(x, CLayers, hLayers, lambda, batch, ImgSize, nclasses, handle);
}

/*
	dim3(1),dim3(batch)
*/
__global__ void g_getCorrect(double* softMaxP, int cols,  int* vote)
{
	int id = threadIdx.x;

	double* p = softMaxP + id * cols;
	int* votep= vote     + id * cols;

	int r = 0;
	double maxele = log(p[0]);
	for(int i = 1; i < cols; i++)
	{
		double val = log(p[i]);
		if(maxele < val)
		{
			maxele = val;
			r = i;
		}
	}
	votep[r]++;
}

void resultProdict(double** testX, int*testY,
	int* vote,
	std::vector<cuCvl> &CLayers, 
	std::vector<cuFll> &hLayers, 
	cuSMR &smr, double lambda, int batch, int ImgSize, int nclasses, cublasHandle_t handle)
{
	convAndPooling(testX, CLayers, batch, ImgSize);
	getHiddenLayerActi(hLayers, handle);
	getSoftMaxP(smr, handle);
	g_getCorrect<<<dim3(1), batch>>>(
		cuSoftMaxP->devData, 
		cuSoftMaxP->cols,
		vote);
	cudaDeviceSynchronize();
}

void gradientChecking(std::vector<cuCvl> &CLayers, std::vector<cuFll> &hLayers, cuSMR &smr, double**x, 
	int*y, double lambda, int batch, int ImgSize, int nclasses, cublasHandle_t handle)
{
	for(int hl = 0; hl < hLayers.size(); hl++)
	{
		dropDelta(hLayers[hl].dropW, Config::instance()->getFC()[hl]->m_dropoutRate);
	}

	std::cout<<"test network !!!!"<<std::endl;
	double epsilon = 1e-4;

	for(int a = 0; a < CLayers.size(); a++)
	{
		for(int b = 0; b < CLayers[a].layer.size(); b++)
		{
			printf("====%d %d\n",a, b);
			getNetworkCost(x,
				y,
				CLayers, hLayers,
				smr,
				lambda, batch, ImgSize, nclasses, handle);

			CLayers[a].layer[b].Wgrad->toCpu();

			cuMatrix<double>* grad = new cuMatrix<double>(CLayers[a].layer[b].Wgrad->hostData, CLayers[a].layer[b].Wgrad->rows,
				CLayers[a].layer[b].Wgrad->cols, CLayers[a].layer[b].Wgrad->channels);

			for(int c = 0; c < CLayers[a].layer[b].W->channels; c++){
				for(int i = 0; i < CLayers[a].layer[b].W->rows; i++){
					for(int j = 0; j < CLayers[a].layer[b].W->cols; j++){
						double memo = CLayers[a].layer[b].W->get(i, j, c);
						CLayers[a].layer[b].W->set(i, j, c, memo + epsilon);
						CLayers[a].layer[b].W->toGpu();
						getNetworkCost(x, y, CLayers, hLayers, smr, lambda, batch, ImgSize, nclasses, handle);
						smr.cost->toCpu();
						double value1 = smr.cost->get(0, 0 , 0);
						CLayers[a].layer[b].W->set(i, j, c, memo - epsilon);
						CLayers[a].layer[b].W->toGpu();
						getNetworkCost(x, y, CLayers, hLayers, smr, lambda, batch, ImgSize, nclasses, handle);
						smr.cost->toCpu();
						double value2 = smr.cost->get(0, 0, 0);
						double tp = (value1 - value2) / (2 * epsilon);
						if(fabs(tp - grad->get(i, j, c)) > 0.00001)
							std::cout<<i<<","<<j<<","<<c<<","<<tp<<", "<<grad->get(i,j,c)<<", "
							<<tp - grad->get(i,j,c)<<std::endl;
						CLayers[a].layer[b].W->set(i, j, c, memo);
						CLayers[a].layer[b].W->toGpu();
					}
				}
			}
			delete grad;
		}

	}
}

/*
	
*/
void __global__ g_getVotingResult(int* voting, int* y, int* correct, int len, int nclasses)
{
	for(int i = 0; i < len; i += blockDim.x * gridDim.x)
	{
		int idx = i + blockDim.x * blockIdx.x + threadIdx.x;
		if(idx < len)
		{
			int* pvoting = voting + idx * nclasses;
			int _max = pvoting[0];
			int rid  = 0;
			for(int j = 1; j < nclasses; j++)
			{
				if(pvoting[j] > _max)
				{
					_max = pvoting[j];
					rid  = j;
				}
			}
			if(rid == y[idx])
			{
				atomicAdd(correct, 1);
			}
		}
	}
}

void predictTestDate(cuMatrixVector<double>&x,
		cuMatrix<int>*y ,
		std::vector<cuCvl> &CLayers,
		std::vector<cuFll> &HiddenLayers,
		cuSMR &smr,
		double lambda,
		cuMatrixVector<double>&testX,
		cuMatrix<int>* testY,
		int nsamples,
		int batch,
		int ImgSize,
		int nclasses,
		cublasHandle_t handle) {

	for (int hl = 0; hl < HiddenLayers.size(); hl++) {
		dropDelta(HiddenLayers[hl].dropW, 0.0);
	}

	cuVote->gpuClear();

	int cropr[] = { Config::instance()->getCrop() / 2, 0, 0,
			Config::instance()->getCrop(), Config::instance()->getCrop() };
	int cropc[] = { Config::instance()->getCrop() / 2, 0,
			Config::instance()->getCrop(), 0, Config::instance()->getCrop() };

//	double scale[] = {0.0, -Config::instance()->getScale(), Config::instance()->getScale()};
//	double rotation[] = {0.0, -Config::instance()->getRotation(), Config::instance()->getRotation()};

//	for (int s = 0; s < (Config::instance()->getScale() <= 0.00001 ? 1 : 3);s++) {
//		for (int r = 0; r < (Config::instance()->getRotation() <= 0.00001 ? 1 : 3); r++) {
			for (int h = 0; h < (Config::instance()->getHorizontal() == true ? 2 : 1); h++) {
				for (int c = 0; c < (Config::instance()->getCrop() == 0 ? 1 : 5); c++) {
					for (int p = 0; p < testX.size() / batch; p++) {
						printf("test  %2d%%", 100 * p / ((testX.size() + batch - 1) / batch));
						int tstart = p * batch;
						cuApplyCrop(testX.m_devPoint + tstart,
								cu_distortion_vector->m_devPoint, batch,
								ImgSize, cropr[c], cropc[c]);

						if (h == 1)
							cuApplyHorizontal(cu_distortion_vector->m_devPoint,
									cu_distortion_vector->m_devPoint, batch,
									ImgSize);

//						cuApplyScaleAndRotate(batch, ImgSize, scale[s], rotation[r]);
//						cuApplyDistortion(cu_distortion_vector->m_devPoint,
//								cu_distortion_vector->m_devPoint, batch, ImgSize);

//						for (int ff = batch - 1; ff >= 0; ff--) {
//							showImg(testX[tstart + ff], 10);
//							showImg(cu_distortion_vector->m_vec[ff], 10);
//							cv::waitKey(0);
//						}

						resultProdict(cu_distortion_vector->m_devPoint,
								testY->devData + tstart,
								cuVote->devData + tstart * nclasses, CLayers,
								HiddenLayers, smr, lambda, batch, ImgSize,
								nclasses, handle);

						printf("\b\b\b\b\b\b\b\b\b");
					}
				}
			}
//		}
//	}

	cuCorrect->gpuClear();
	g_getVotingResult<<<dim3((testX.size() + batch - 1) / batch), dim3(batch)>>>(
			cuVote->devData,
			testY->devData,
			cuCorrect->devData,
			testX.size(),
			nclasses);
	cudaDeviceSynchronize();
	getLastCudaError("g_getVotingResult");
	cuCorrect->toCpu();

	if (cuCorrect->get(0, 0, 0) > cuCurCorrect) {
		cuCurCorrect = cuCorrect->get(0, 0, 0);
		cuSaveConvNet(CLayers, HiddenLayers, smr, ImgSize, nclasses, nsamples);
	}
}

void cuTrainNetwork(cuMatrixVector<double>&x,
	cuMatrix<int>*y ,
	std::vector<cuCvl> &CLayers,
	std::vector<cuFll> &HiddenLayers,
	cuSMR &smr,
	double lambda,
	cuMatrixVector<double>&testX,
	cuMatrix<int>* testY,
	int nsamples,
	int batch,
	int ImgSize,
	int nclasses,
	cublasHandle_t handle)
{
 	if(Config::instance()->getIsGradientChecking())
 		gradientChecking(CLayers, HiddenLayers, smr, x.m_devPoint, y->devData, lambda, batch, ImgSize, nclasses, handle);

 	predictTestDate(x,
 			y,
 			CLayers,
 			HiddenLayers,
 			smr,
 			lambda,
 			testX,
 			testY,
 			nsamples,
 			batch,
 			ImgSize,
 			nclasses,
 			handle);

 	printf("correct is %d\n", cuCorrect->get(0,0,0));
	
	int epochs = 20000;
 	double nlrate[] =    {0.05, 0.04, 0.03, 0.02, 0.01, 0.008, 0.006, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 
		0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125, 0.000390625, 0.0001953125, 0.00009765625, 0.000048828125, 0.0000244140625};
 	double nMomentum[] = {0.90, 0.91, 0.92, 0.93, 0.94, 0.942, 0.944, 0.95,    0.96,   0.97,  0.995,  0.90,  
		0.92,  0.93,   0.94,    0.95,     0.96,      0.97,       0.98,        0.99,         0.994,         0.995,          0.996};
 	int epoCount[] =     {50,   50,     50,   50,   50,    50,    50,   50,      50,     50,     50,   50,   
		50,    50,     50,      50,       50,        50,         50,          50,           50,            50,             50};

	while(1)
	{
		double lrate;
		double Momentum;
		int id = 0;
		for(int epo = 0; epo < epochs; epo++){
			if(id >= sizeof(nlrate) / sizeof(double))
				break;
			lrate = nlrate[id];
			Momentum = nMomentum[id];
	
			double start,end;
			start = clock();

			cuApplyRandom(batch, clock(), ImgSize);
			for(int hl = 0; hl < HiddenLayers.size(); hl++)
			{
				dropDelta(HiddenLayers[hl].dropW, Config::instance()->getFC()[hl]->m_dropoutRate);
			}
			for(int k = 0; k < (x.size() + batch - 1) / batch; k++)
			{
				int start = rand() % (x.size() - batch);
				//int start = k * batch;
				printf("train %2d%%", 100 * k / ((x.size() + batch - 1) / batch));
				cuApplyCropRandom(x.m_devPoint + start, cu_distortion_vector->m_devPoint, batch, ImgSize);
				cuApplyDistortion(cu_distortion_vector->m_devPoint, cu_distortion_vector->m_devPoint, batch, ImgSize);

				if(Config::instance()->getHorizontal() && epo % 2 == 1){
					cuApplyHorizontal(cu_distortion_vector->m_devPoint, cu_distortion_vector->m_devPoint, batch, ImgSize);
				}

				if(Config::instance()->getImageShow())
				{
					for(int ff = batch - 1; ff >= 0; ff--)
					{
						showImg(x[start + ff], 10);
						showImg(cu_distortion_vector->m_vec[ff], 10);
						cv::waitKey(0);
					}
				}
				
				getNetworkCost(cu_distortion_vector->m_devPoint,
					y->devData + start,
					CLayers, HiddenLayers,
					smr,
					lambda, batch, ImgSize, nclasses, handle);
				updataWB(CLayers, HiddenLayers, smr, lrate, Momentum, batch);
				printf("\b\b\b\b\b\b\b\b\b");
			}

			smr.cost->toCpu();
			char str[512];

				predictTestDate(x,
			 			y,
			 			CLayers,
			 			HiddenLayers,
			 			smr,
			 			lambda,
			 			testX,
			 			testY,
			 			nsamples,
			 			batch,
			 			ImgSize,
			 			nclasses,
			 			handle);

			if(epo && epo % epoCount[id] == 0)
			{
				cu_v_smr_w->gpuClear();
				cu_v_smr_b->gpuClear();
				for(int i = 0; i < HiddenLayers.size(); i++)
				{
					cu_v_hl_w[i]->gpuClear();
					cu_v_hl_b[i]->gpuClear();
				}
				for(int cl = 0; cl < CLayers.size(); cl++)
				{
					for(int i = 0; i < CLayers[cl].layer.size(); i++)
					{
						cu_v_cvl_w[cl][i]->gpuClear();
						cu_v_cvl_b[cl][i]->gpuClear();
					}
				}
				id++;
			}
			end = clock();

			sprintf(str, "e=%d t=%.03lfs cst=%lf crt=%d/%d mom=%.06lf r=%.08lf", epo,
				(double)(end - start) / CLOCKS_PER_SEC, smr.cost->get(0,0,0), cuCorrect->get(0, 0, 0), cuCurCorrect, Momentum, lrate);

			printf("%s\n",str);
			LOG(str,"log.txt");
		}
	}
}
