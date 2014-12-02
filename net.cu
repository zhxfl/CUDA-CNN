#include "net.cuh"
#include "opencv2/opencv.hpp"
#include "cuMatrix.h"
#include <cuda_runtime.h>
#include "util.h"
#include <time.h>
#include "cuDistortion.cuh"
#include "Config.h"
#include "cuMatrixVector.h"

cuMatrixVector<double>* cu_distortion_vector;

std::vector<int> cuPoolOutputSize;
std::vector<int> cuConvOutputSize;
std::vector<int> cuKernelScan;

//卷积层输出
std::vector<cuMatrix<int>*>cuPointX;
std::vector<cuMatrix<int>*>cuPointY;
std::vector<cuMatrix<double>*>cuConv;
std::vector<cuMatrix<double>*>cuPool;

//隐藏层的输出
std::vector<cuMatrix<double>*>cuHiddenActi;

//回归层的输出
cuMatrix<double>* cuSoftMaxP;
cuMatrix<double>* cuGroundTruth;

//反向传导
cuMatrix<double>* cuSoftMaxDelta;
std::vector<cuMatrix<double>*>cuHiddenDelta;
std::vector<cuMatrix<double>*>cuConvDelta;
std::vector<cuMatrix<double>*>cuPoolDelta;
std::vector<cuMatrix<double>*>cuPoolDeltaAndBorder;
std::vector<cuMatrix<double>*>cuConvLayerWgradTmp;

//势
cuMatrix<double>* cu_v_smr_w;
cuMatrix<double>* cu_v_smr_b;
std::vector<cuMatrix<double>*>cu_v_hl_w;
std::vector<cuMatrix<double>*>cu_v_hl_b;
std::vector<std::vector<cuMatrix<double>*> >cu_v_cvl_w;
std::vector<std::vector<cuMatrix<double>*> >cu_v_cvl_b;
int cuCurCorrect;

//正确的个数
cuMatrix<double>* cuCorrect;
cuMatrix<double>* cuCorrectCount = NULL;

/*
	函数功能：卷积层有多个核，在GPU做并发时，
			  内存比较散乱，这里讲这些核的参数组织成
			  二维数组方便并发访问
*/
void cuConvLayer::init()
{
	cudaError_t cudaStat;
	//w
	h_w = (double**)malloc(sizeof(double*) * layer.size());
	if(!h_w)
	{
		printf("cuConvLayer::init malloc h_w\n");
		return;
	}
	cudaStat = cudaMalloc((void**)&d_w, sizeof(double*) * layer.size());
	if(cudaStat != cudaSuccess){
		printf("cuConvLayer::init cudaMalloc h_w fail\n");
		return;
	}

	for(int i = 0; i < layer.size(); i++){
		h_w[i] = layer[i].W->devData;
	}

	cudaStat = cudaMemcpy(d_w, h_w, sizeof(double*) * layer.size(), cudaMemcpyHostToDevice);
	if(cudaStat != cudaSuccess){
		printf("cuConvLayer::init cudaMemcpy w fail\n");
		return;
	}

	//b
	h_b = (double**)malloc(sizeof(double*) * layer.size());
	if(!h_b)
	{
		printf("cuConvLayer::init malloc h_b fail\n");
		return;
	}
	cudaStat = cudaMalloc((void**)&d_b, sizeof(double*) * layer.size());
	if(cudaStat != cudaSuccess){
		printf("cuConvLayer::init cudaMalloc h_b fail\n");
		return;
	}

	for(int i = 0; i < layer.size(); i++){
		h_b[i] = layer[i].b->devData;
	}

	cudaStat = cudaMemcpy(d_b, h_b, sizeof(double*) * layer.size(), cudaMemcpyHostToDevice);
	if(cudaStat != cudaSuccess){
		printf("cuConvLayer::init cudaMemcpy b fail\n");
		return;
	}

	//wgrad
	h_wgrad = (double**)malloc(sizeof(double*) * layer.size());
	if(!h_wgrad){
		printf("cuConvLayer::init malloc h_wgrad fail\n");
		return;
	}

	cudaStat = cudaMalloc((void**)&d_wgrad, sizeof(double*) * layer.size());
	if(cudaStat != cudaSuccess){
		printf("cuConvLayer::init cudaMalloc wgrad fail\n");
		return;
	}

	for(int i = 0; i < layer.size(); i++){
		h_wgrad[i] = layer[i].Wgrad->devData;
	}

	cudaStat = cudaMemcpy(d_wgrad, h_wgrad, sizeof(double*) * layer.size(), cudaMemcpyHostToDevice);
	if(cudaStat != cudaSuccess){
		printf("cuConvLayer::init cudaMemcpy wgrad fail\n");
		return;
	}

	//bgrad
	h_bgrad = (double**)malloc(sizeof(double*) * layer.size());
	if(!h_bgrad){
		printf("cuConvLayer::init malloc h_bgrad fail\n");
	}
	cudaStat = cudaMalloc((void**)&d_bgrad, sizeof(double*) * layer.size());
	if(cudaStat != cudaSuccess){
		printf("cuConvLayer::init cudaMalloc bgrad fail\n");
		return;
	}

	for(int i = 0; i < layer.size(); i++){
		h_bgrad[i] = layer[i].bgrad->devData;
	}

	cudaStat = cudaMemcpy(d_bgrad, h_bgrad, sizeof(double*) * layer.size(), cudaMemcpyHostToDevice);
	if(cudaStat != cudaSuccess){
		printf("cuConvLayer::init cudaMemcpy bgrad fail\n");
		return;
	}
}

void createGaussian(double* gaussian, double dElasticSigma1, double dElasticSigma2, int rows, int cols, double epsilon)
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
	//printf("\n\n");
	//exit(0);
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
	cv::Mat ran = cv::Mat::zeros(M->rows, M->cols, CV_64FC1);
	cv::theRNG().state = clock();
	randu(ran, cv::Scalar(0), cv::Scalar(1.0));
	for(int i = 0; i < ran.rows; i++)
	{
		for(int j = 0; j < ran.cols; j++)
		{
			double r = ran.at<double>(i,j);
			if(r < cuDropProb)
				M->set(i, j, 0.0);
			else 
				M->set(i, j, 1.0);
			//printf("%lf ", ntw.M->get(i,j));
		}//printf("\n");
	}
	M->toGpu();
}

/*
函数功能：卷积层参数初始化
参    数：
ConvK &convk  卷积层
int width     大小
*/

void weightRandomInit(cuConvK &convk, int width){
	double epsilon = 0.1f;
	convk.W = new cuMatrix<double>(width, width);
	double r1 = 0.5 + 4.0 * (rand()) / RAND_MAX;
	double r2 = 0.5 + 4.0 * (rand()) / RAND_MAX;
	//printf("%f %f\n",r1, r2);
	createGaussian(convk.W->hostData, r1,r2, width, width, epsilon * 0.5 + epsilon * rand() / RAND_MAX);

// 	for(int i = 0; i<convk.W->rows; i++){
// 		for(int j=0; j<convk.W->cols; j++){
// 			convk.W->set(i, j, 1.0 * rand() / RAND_MAX * 2.0 * epsilon - epsilon);
// 		}
// 	}
	convk.W->toGpu();
	convk.b = new cuMatrix<double>(1, 1);
	convk.Wgrad = new cuMatrix<double>(width, width);
	convk.bgrad = new cuMatrix<double>(1, 1);
}


/*
函数功能：隐藏层参数初始化
参    数：
Ntw &ntw 隐藏层
int inputsize  每个神经元链接上一层神经元个数 
int hiddensize 本层的神经元个数
*/

void weightRandomInit(cuNtw &ntw, int inputsize, int hiddensize, double dropRate){
	double epsilon = sqrt((double)6) / sqrt((double)(hiddensize + inputsize + 1));
	ntw.W = new cuMatrix<double>(hiddensize, inputsize);
	ntw.dropW = new cuMatrix<double>(hiddensize, inputsize);
	ntw.afterDropW = new cuMatrix<double>(hiddensize, inputsize);

	//double r1 = 2.0 + 4.0 * (rand()) / RAND_MAX;
	//double r2 = 2.0 + 4.0 * (rand()) / RAND_MAX;
	//createGaussian(ntw.W->hostData, r1, r2, ntw.W->rows, ntw.W->cols, epsilon * 0.5 + epsilon * rand() / RAND_MAX);

	for(int i=0; i < ntw.W->rows; i++){
		for(int j=0; j< ntw.W->cols; j++){
			ntw.W->set(i,j, 1.0 * rand() / RAND_MAX *  2.0 * epsilon - epsilon);
		}
	}
	dropDelta(ntw.dropW, dropRate);
	ntw.W->toGpu();
	ntw.b = new cuMatrix<double>(hiddensize, 1);
	ntw.Wgrad = new cuMatrix<double>(hiddensize, inputsize);
	ntw.bgrad = new cuMatrix<double>(hiddensize, 1);
}

/*
函数功能：softMax层神经元初始化
参    数：
SMR &smr      softMax层
int nclasses  输出类别个数
int nfeatures 输入
*/
void weightRandomInit(cuSMR &smr, int nclasses, int nfeatures){
	double epsilon = 0.01f;
	smr.Weight = new cuMatrix<double>(nclasses, nfeatures);
	for(int i = 0; i<smr.Weight->rows; i++){
		for(int j=0; j<smr.Weight->cols; j++){
			smr.Weight->set(i,j, 1.0 * rand() / RAND_MAX *  2.0 * epsilon - epsilon);     
		}
	}
	smr.Weight->toGpu();

	smr.b = new cuMatrix<double>(nclasses, 1);
	smr.cost = new cuMatrix<double>(1, 1);
	smr.Wgrad = new cuMatrix<double>(nclasses, nfeatures);
	smr.bgrad = new cuMatrix<double>(nclasses, 1);
}

void cuConvNetInitPrarms(std::vector<cuCvl> &ConvLayers,
	std::vector<cuNtw> &HiddenLayers,
	cuSMR &smr,
	int imgDim,
	int nsamples, 
	int nclasses)
{
	srand(time(NULL));
	// Init Conv layers
	for(int i=0; i < Config::instance()->getConv().size(); i++){
		cuCvl tpcvl;
		for(int j=0; j < Config::instance()->getConv()[i]->m_amount; j++){
			cuConvK tmpConvK;
			weightRandomInit(tmpConvK, Config::instance()->getConv()[i]->m_kernelSize);
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
	cuNtw tpntw;
	weightRandomInit(tpntw, hiddenfeatures, Config::instance()->getFC()[0]->m_numHiddenNeurons, 
		Config::instance()->getFC()[0]->m_dropoutRate);
	HiddenLayers.push_back(tpntw);
	for(int i=1; i < Config::instance()->getFC().size(); i++){
		cuNtw tpntw2;
		weightRandomInit(tpntw2, Config::instance()->getFC()[i - 1]->m_numHiddenNeurons,
			Config::instance()->getFC()[i]->m_numHiddenNeurons, Config::instance()->getFC()[i]->m_dropoutRate);
		HiddenLayers.push_back(tpntw2);
	}
	// Init Softmax layer
	weightRandomInit(smr, nclasses, Config::instance()->getFC()[Config::instance()->getFC().size() - 1]->m_numHiddenNeurons);
}

void saveWeight(cuConvK &convk, int width, FILE*pOut)
{
	convk.W->toCpu();
	convk.b->toCpu();
	for(int i = 0; i<convk.W->rows; i++){
		for(int j=0; j<convk.W->cols; j++){
			fprintf(pOut, "%lf ",convk.W->get(i,j));      
		}
	}
	fprintf(pOut, "%lf ", convk.b->get(0,0));
}

void saveWeight(cuNtw &ntw, int inputsize, int hiddensize, FILE*pOut){
	ntw.W->toCpu();
	ntw.b->toCpu();
	for(int i=0; i<ntw.W->rows; i++){
		for(int j=0; j<ntw.W->cols; j++){
			fprintf(pOut, "%lf ", ntw.W->get(i,j));
		}
	}

	for(int i = 0; i < ntw.b->rows; i++){
		for(int j = 0; j < ntw.b->cols;  j++){
			fprintf(pOut, "%lf ", ntw.b->get(i,j));
		}
	}
}

void saveWeight(cuSMR &smr, int nclasses, int nfeatures, FILE* pOut){
	smr.Weight->toCpu();
	smr.b->toCpu();
	for(int i = 0; i<smr.Weight->rows; i++){
		for(int j=0; j<smr.Weight->cols; j++){
			fprintf(pOut, "%lf ", smr.Weight->get(i,j)); 
		}
	}

	for(int i = 0; i < smr.b->rows; i++){
		for(int j = 0; j < smr.b->cols;  j++){
			fprintf(pOut, "%lf ", smr.b->get(i,j));
		}
	}
}

void cuSaveConvNet(std::vector<cuCvl> &ConvLayers,
	std::vector<cuNtw> &HiddenLayers,
	cuSMR &smr,
	int imgDim,
	int nclasses,
	int nsamples)
{	
	FILE *pOut = fopen("net.txt", "w");
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

	cuNtw tpntw = HiddenLayers[0];
	saveWeight(tpntw, hiddenfeatures, Config::instance()->getFC()[0]->m_numHiddenNeurons, pOut);
	for(int i=1; i < Config::instance()->getFC().size(); i++){
		cuNtw tpntw2 = HiddenLayers[i];
		saveWeight(tpntw2, Config::instance()->getFC()[i - 1]->m_numHiddenNeurons, 
			Config::instance()->getFC()[i]->m_numHiddenNeurons, pOut);
	}
	// Init Softmax layer
	saveWeight(smr, nclasses, Config::instance()->getFC()[Config::instance()->getFC().size() - 1]->m_numHiddenNeurons, pOut);
	fclose(pOut);
};

void readWeight(cuConvK &convk, int width, FILE*pIn)
{
	convk.W = new cuMatrix<double>(width, width); 
	double val = 0.0;
	for(int i = 0; i<convk.W->rows; i++){
		for(int j=0; j<convk.W->cols; j++){
			fscanf(pIn, "%lf", &val); 
			convk.W->set(i,j,val);
		}
	}
	fscanf(pIn, "%lf ", &val);
	convk.b = new cuMatrix<double>(1, 1);
	convk.b->set(0,0,val);
	convk.Wgrad = new cuMatrix<double>(width, width);
	convk.bgrad = new cuMatrix<double>(1, 1);
	convk.W->toGpu();
	convk.b->toGpu();
}

void readWeight(cuNtw &ntw, int inputsize, int hiddensize, FILE*pIn, double dropRate){
	double val = 0.0;
	ntw.W = new cuMatrix<double>(hiddensize, inputsize);
	for(int i=0; i<ntw.W->rows; i++){
		for(int j=0; j<ntw.W->cols; j++){
			fscanf(pIn, "%lf", &val);
			ntw.W->set(i,j,val);
		}
	}

	ntw.dropW = new cuMatrix<double>(hiddensize, inputsize);
	dropDelta(ntw.dropW, dropRate);


	ntw.b = new cuMatrix<double>(hiddensize, 1);
	for(int i = 0; i < ntw.b->rows; i++){
		for(int j = 0; j < ntw.b->cols; j++){
			fscanf(pIn, "%lf", &val);
			ntw.b->set(i,j,val);
		}
	}
	
	ntw.Wgrad = new cuMatrix<double>(hiddensize, inputsize);
	ntw.bgrad = new cuMatrix<double>(hiddensize, 1);
	ntw.W->toGpu();
	ntw.b->toGpu();
}

void readWeight(cuSMR &smr, int nclasses, int nfeatures, FILE* pIn){
	smr.Weight = new cuMatrix<double>(nclasses, nfeatures);
	double val = 0.0;
	for(int i = 0; i<smr.Weight->rows; i++){
		for(int j=0; j<smr.Weight->cols; j++){
			fscanf(pIn, "%lf", &val);
			smr.Weight->set(i,j,val);
		}
	}
	smr.b = new cuMatrix<double>(nclasses, 1);
	for(int i = 0; i < smr.b->rows; i++){
		for(int j = 0; j < smr.b->cols; j++){
			fscanf(pIn, "%lf ", &val);
			smr.b->set(i,j,val);
		}
	}
	smr.cost = new cuMatrix<double>(1,1);
	smr.Wgrad = new cuMatrix<double>(nclasses, nfeatures);
	smr.bgrad = new cuMatrix<double>(nclasses, 1);
	smr.Weight->toGpu();
	smr.b->toGpu();
}

void cuFreeConvNet(std::vector<cuCvl> &ConvLayers,
	std::vector<cuNtw> &HiddenLayers,
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
	std::vector<cuNtw> &HiddenLayers,
	cuSMR &smr, int imgDim, int nsamples, char* path,
	int nclasses)
{	
	FILE *pIn = fopen(path, "r");
	// Init Conv layers
	// 这里虽然没有浅拷贝和泄露的问题，但是在内存管理上依然存在隐患
	// cuConvk是一个临时变量,系统会释放他(new出来的没有释放)
	// 但是push_back新建一个对象又用到了之前没有释放的空间，所以刚好没有泄露
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

	cuNtw tpntw;
	readWeight(tpntw, hiddenfeatures, Config::instance()->getFC()[0]->m_numHiddenNeurons,
		pIn, Config::instance()->getFC()[0]->m_dropoutRate);
	HiddenLayers.push_back(tpntw);
	for(int i=1; i < Config::instance()->getFC().size(); i++){
		cuNtw tpntw2;
		readWeight(tpntw2, Config::instance()->getFC()[i - 1]->m_numHiddenNeurons, 
			Config::instance()->getFC()[i]->m_numHiddenNeurons, pIn, Config::instance()->getFC()[i]->m_dropoutRate);
		HiddenLayers.push_back(tpntw2);
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
	std::vector<cuNtw>& HiddenLayers,
	cuSMR& smr,
	int ImgSize,
	int nclasses)
{
	cudaError_t cudaStat;
	//////////////
	//卷积层的输出
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

		cuConv.push_back  (new cuMatrix<double>(batch, curKernelAmount * curSize * curSize));
		curSize = (curSize + Config::instance()->getConv()[i]->m_poolingDim - 1) / Config::instance()->getConv()[i]->m_poolingDim;
		cuPool.push_back  (new cuMatrix<double>(batch, curKernelAmount * curSize * curSize));
		cuPointX.push_back(new cuMatrix<int>(batch, curKernelAmount * curSize * curSize));
		cuPointY.push_back(new cuMatrix<int>(batch, curKernelAmount * curSize * curSize));
	}

	//////////////
	//隐藏层的输出
	//////////////
	for(int i = 0; i < HiddenLayers.size(); i++)
	{
		cuHiddenActi.push_back(new cuMatrix<double>(batch, Config::instance()->getFC()[i]->m_numHiddenNeurons));
	}

	//////////////
	//回归层的输出
	//////////////
	cuSoftMaxP    = new cuMatrix<double>(batch, nclasses);
	cuGroundTruth = new cuMatrix<double>(batch, nclasses);

	/////////////
	//反向传导
	/////////////
	cuSoftMaxDelta = new cuMatrix<double>(batch, nclasses);
	for(int i = 0; i < HiddenLayers.size(); i++)
	{
		cuHiddenDelta.push_back(new cuMatrix<double>(batch, Config::instance()->getFC()[i]->m_numHiddenNeurons));
	}

	/////////
	//卷积层
	/////////
	for(int i = 0; i < cuPool.size(); i++)
	{
		cuPoolDelta.push_back(new cuMatrix<double>(cuPool[i]->rows, cuPool[i]->cols));
	}
	for(int i = 0; i < cuConv.size(); i++)
	{
		cuConvDelta.push_back(new cuMatrix<double>(cuConv[i]->rows, cuConv[i]->cols));
	}
	for(int cl = ConvLayers.size() - 1; cl > 0; cl--)
	{
		int size = cuConvOutputSize[cl] + Config::instance()->getConv()[cl]->m_kernelSize - 1;
		cuPoolDeltaAndBorder.push_back(new cuMatrix<double>(batch, cuKernelScan[cl]
		* size * size));
	}
	for(int cl = 0; cl < ConvLayers.size(); cl++)
	{
		cuConvLayerWgradTmp.push_back(new cuMatrix<double>(batch,
			cuKernelScan[cl] * + Config::instance()->getConv()[cl]->m_kernelSize * Config::instance()->getConv()[cl]->m_kernelSize));
	}

	//势能
	cu_v_smr_w = new cuMatrix<double>(smr.Weight->rows, smr.Weight->cols);
	cu_v_smr_b = new cuMatrix<double>(smr.b->rows, smr.b->cols);
	for(int i = 0; i < HiddenLayers.size(); i++)
	{
		cu_v_hl_w.push_back(new cuMatrix<double>(HiddenLayers[i].W->rows, HiddenLayers[i].W->cols));
		cu_v_hl_b.push_back(new cuMatrix<double>(HiddenLayers[i].b->rows, HiddenLayers[i].b->cols));
	}

	for(int cl = 0; cl < ConvLayers.size(); cl++)
	{
		std::vector<cuMatrix<double>*>tmpVecW;
		std::vector<cuMatrix<double>*>tmpVecb;
		for(int i = 0; i < ConvLayers[cl].layer.size(); i++)
		{
			cuMatrix<double>* tmpW = new cuMatrix<double>(ConvLayers[cl].layer[i].W->rows, 
				ConvLayers[cl].layer[i].W->cols);
			cuMatrix<double>* tmpb = new cuMatrix<double>(1,1);
			tmpVecW.push_back(tmpW);
			tmpVecb.push_back(tmpb);
		}
		cu_v_cvl_w.push_back(tmpVecW);
		cu_v_cvl_b.push_back(tmpVecb);
	}

	//代价
	if(cuCorrectCount == NULL)
	{
		cuCorrectCount = new cuMatrix<double>(60000, nclasses);
		cuCorrect = new cuMatrix<double>(1,1);
	}

	//畸变之后的数据
	cu_distortion_vector = new cuMatrixVector<double>();
	for(int i = 0; i < batch; i++){
		cu_distortion_vector->m_vec.push_back(new cuMatrix<double>(ImgSize, ImgSize));
	}
	cu_distortion_vector->toGpu();
}

void cuFreeCNNMemory(
	int batch,
	cuMatrixVector<double>&trainX, 
	cuMatrixVector<double>&testX,
	std::vector<cuCvl>&ConvLayers,
	std::vector<cuNtw> &HiddenLayers, 
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

	//势能
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
	//代价

	//畸变之后的数据
	delete cu_distortion_vector;
}

/*
	函数说明   ：第一个卷积层和池化层前向传导的GPU函数
	arrayS     : 输入的样本数据保存为二维数组
	arrayW     : 第一个卷积层的网络权值
	arrayB     : 第一个卷积层的偏置值
	conv       : 卷积层输出
	pool       : 池化层输出
	pointX	   : 池化之后所选中的x坐标记录，在反向传导中使用
	pointY     : 池化之后所选中的y坐标记录，在反向传导中使用
	inputSize  : 输入样本的尺寸
	kernelSize : 卷积核的尺寸
	convSize   : 卷积层输出之后的尺寸
	poolSize   : 池化层输出之后的尺寸
	poolingDim : 池化层的跳步（每个大小为poolingDim X PoolingDim的图片选取一个最大值）
	k1Amount   : 第一个卷积层的卷积核个数（默认为7）


	并行线程分配：
	<<<dim3(batch, cuKernelAmount[0]),dim3(convSize, convSize)>>>
	这个并行分配需要注意convSize X convSize <= 1024，所以实际上图片不能输入过大，
	在手写数字数据集上，convSize = 24不会有问题。
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
	int k1Amount,
	int NONLIN)
{
	int sp = blockIdx.x;
	int k  = blockIdx.y;
	int x  = threadIdx.y;
	int y  = threadIdx.x;

	int convSize2 = convSize * convSize;
	int poolSize2 = poolSize * poolSize;
	int convSkip  = sp * k1Amount * convSize2 + k * convSize2;
	int poolSkip  = sp * k1Amount * poolSize2 + k * poolSize2;

	double* curInput = arrayS[sp];
	double* w        = arrayW[k];
	double  b        = arrayB[k][0];

	double* curConv  = conv   + convSkip;
	double* curPool  = pool   + poolSkip;
	int* px          = pointX + poolSkip;
	int* py          = pointY + poolSkip;
	
	//卷积计算
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
	curConv[x * convSize + y] = d_nonLinearity(val + b, NONLIN);
	__syncthreads();

	if(x >= poolSize)
		return;
	if(y >= poolSize)
		return;

	int curX = x * poolingDim;
	int curY = y * poolingDim;

	double _max = curConv[curX * convSize + curY];
	int lenx = min(convSize, (x + 1) * poolingDim);
	int leny = min(convSize, (y + 1) * poolingDim);

	for(int i = x * poolingDim; i < lenx; i++)
	{
		for(int j = y * poolingDim; j < leny; j++)
		{
			val = curConv[i * convSize + j];
			if(_max < val){
				_max  = val;
				curX = i;
				curY = j;
			}
		}
	}
	
	int idx = x * poolSize + y;
	px     [idx] = curX;
	py     [idx] = curY;
	curPool[idx] = _max;
}

/*
	函数说明   ：第二个（或者之后）卷积层和池化层前向传导的GPU函数
	pool1      : 第一个卷积层池化之后的数据，作为第二个卷积层的输入
	arrayW     : 第二个卷积层的网络权值
	arrayB     : 第二个卷积层的偏置值
	conv2      : 第二个卷积层输出
	pool2      : 第二个池化层输出
	pointX	   : 池化之后所选中的x坐标记录，在反向传导中使用
	pointY     : 池化之后所选中的y坐标记录，在反向传导中使用
	pool1Size  : 输入样本的尺寸
	kernelSize : 卷积核的尺寸
	conv2Size  : 卷积层输出之后的尺寸
	pool2Size  : 池化层输出之后的尺寸
	poolingDim : 池化层的跳步（每个大小为poolingDim X PoolingDim的图片选取一个最大值）
	k1Amount   : 第一个卷积层的卷积核个数（默认为7）
	k2Amount   : 第二个卷积层的卷积核个数（默认为9）


	并行线程分配：
	<<<dim3(batch, k2Amount),dim3(conv2Size * conv2Size, k1Amount)>>>
	注意检查conv2Size * conv2Size * k2Amount <= MAX_THREADS
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
	int NONLIN,
	int len)
{
	int id = blockIdx.y * blockDim.y + threadIdx.y;
	if(id >= len) return;

	int sp = blockIdx.x;
	int k2 = id % k2Amount;
	int k1 = id / k2Amount;
	int x  = threadIdx.x / conv2Size;
	int y  = threadIdx.x % conv2Size;
  	double* w   = arrayW[k2];
  	double  b   = arrayB[k2][0];
	int pool1Size2 = pool1Size * pool1Size;
	int conv2Size2 = conv2Size * conv2Size;
	int pool2Size2 = pool2Size * pool2Size;
	int pool2skip2 = sp * k2Scan* pool2Size2
		+ k1 * k2Amount * pool2Size2
		+ k2 * pool2Size2;
  	double* pl1 = pool1 + sp * k1Scan * pool1Size2 
  				+ k1 * pool1Size2;
  	double* cv2 = conv2 + sp * k2Scan * conv2Size2 
  		        + k1 * k2Amount * conv2Size2
  				+ k2 * conv2Size2;
  	double* pl2 = pool2 + pool2skip2;
  	int   * px  = pointX+ pool2skip2;
  	int   * py  = pointY+ pool2skip2;
  
  	//卷积计算
  	double val = 0.0;
  	for(int i = 0; i < kernelSize; i++)
  	{
  		for(int j = 0; j < kernelSize; j++)
  		{
  			int xx = x + i;
  			int yy = y + j;
  			val += pl1[xx * pool1Size + yy] * w[i * kernelSize + j];
  		}
  	}
  	cv2[x * conv2Size + y] = d_nonLinearity(val + b, NONLIN);
  	__syncthreads();
  
  	if(x >= pool2Size)
  		return;
  	if(y >= pool2Size)
  		return;
  
  	int curX = x * poolingDim;
  	int curY = y * poolingDim;
  
  	double _max = cv2[curX * conv2Size + curY];
	int lenx = min(conv2Size, (x + 1) * poolingDim);
	int leny = min(conv2Size, (y + 1) * poolingDim);
  	for(int i = x * poolingDim; i < lenx; i++)
  	{
  		for(int j = y * poolingDim; j < leny; j++)
  		{
  			val = cv2[i * conv2Size + j];
  			if(_max < val)
			{
  				_max  = val;
  				curX = i;
  				curY = j;
  			}
  		}
  	}
  
  	int idx = x * pool2Size + y;
  	px [idx] = curX;
  	py [idx] = curY;
  	pl2[idx] = _max;
}

void outputPoints(cuMatrix<int>* p)
{
	p->toCpu();
	for(int i = 0; i < p->rows; i++)
	{
		for(int j = 0; j < p->cols; j++)
		{
			printf("%d ", p->get(i,j));
		}printf("\n");
	}
}
void outputMatrix(cuMatrix<double>* m)
{
	m->toCpu();
	for(int i = 0; i < m->rows; i++)
	{
		for(int j = 0; j < m->cols; j++)
		{
			printf("%.10lf ", m->get(i,j));
		}printf("\n");
	}
}

void convAndPooling(double** x, std::vector<cuCvl> &CLayers, int batch, int ImgSize)
{
	//第一层
	int curSize = ImgSize - Config::instance()->getConv()[0]->m_kernelSize + 1;//24
	//int inputSize = ImgSize;//28
	int outputSize = curSize;//24

	if(outputSize * outputSize> MAX_THREADS){
		printf("g_convAndPooling_1 > MAX_THREADS\n");
		exit(0);
	}

	//outputMatrix(CLayers[0].layer[0].W);

	g_convAndPooling_1<<<dim3(batch, Config::instance()->getConv()[0]->m_amount), 
		dim3(outputSize, outputSize)>>>(
		x,
		CLayers[0].d_w,
		CLayers[0].d_b,
		cuConv[0]->devData,
		cuPool[0]->devData,
		cuPointX[0]->devData,
		cuPointY[0]->devData,
		ImgSize,
		Config::instance()->getConv()[0]->m_kernelSize,
		cuConvOutputSize[0],
		cuPoolOutputSize[0],
		Config::instance()->getConv()[0]->m_poolingDim,
		Config::instance()->getConv()[0]->m_amount,
		Config::instance()->getNonLinearity()
	);

	cudaDeviceSynchronize();

	for(int i = 1; i < Config::instance()->getConv().size(); i++)
	{
		int len = cuKernelScan[i - 1] * Config::instance()->getConv()[i]->m_amount;
		int threadidx = cuConvOutputSize[i] * cuConvOutputSize[i];
		int threadidy = min(512, len + threadidx - 1) / (threadidx);
		int blockidy  = (len + threadidy - 1) / threadidy;

		if(threadidy * threadidx > MAX_THREADS){
			printf("g_convAndPooling_2 > MAX_THREADS\n");
			exit(0);
		}

		g_convAndPooling_2<<<dim3(batch, blockidy),
			dim3(threadidx, threadidy)>>>
			(cuPool[i - 1]->devData,
			CLayers[i].d_w,
			CLayers[i].d_b,
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
			Config::instance()->getNonLinearity(),
			len);
		cudaDeviceSynchronize();
	}
}

/*
	函数功能：隐藏激活函数
	acti    ：输入
	b       ：偏置值

	线程分配<<<dim3(batch), dim3(NumHiddenNeurons)>>>
	batch为样本的个数，NumHiddenNeurons为当前隐层神经元的个数
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

void getHiddenLayerActi(std::vector<cuNtw>&hLayers, cublasHandle_t handle)
{
	for(int hl = 0; hl < Config::instance()->getFC().size(); hl++)
	{
		g_dropW<<<1, 256>>>(hLayers[hl].W->devData,
			hLayers[hl].dropW->devData, 
			hLayers[hl].afterDropW->devData, 
			hLayers[hl].W->getLen());
		cudaDeviceSynchronize();

		if(hl == 0)
		{
			matrixMulTB(cuPool[cuPool.size() - 1], 
				hLayers[hl].afterDropW, cuHiddenActi[hl], handle);

			if(cuHiddenActi[hl]->cols > MAX_THREADS){
				printf("g_hiddenLayerActi > MAX_THREADS\n");
				exit(0);
			}

			g_hiddenLayerActi<<<cuHiddenActi[hl]->rows, cuHiddenActi[hl]->cols>>>(cuHiddenActi[hl]->devData, 
				hLayers[hl].b->devData, Config::instance()->getNonLinearity());
			cudaDeviceSynchronize();
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
		}
	}
}

/*
	函数功能：计算回归层的输出
	softMaxp: 回归层数据存放处
	b       : 回归层的偏置量
	线程分配：<<<cuSoftMaxP->rows, cuSoftMaxP->cols>>>
*/
__global__ void g_getSoftMaxP(double* softMaxP, double* b)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	__shared__ double _max[10];
	__shared__ double _sum[10];

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

	g_getSoftMaxP<<<cuSoftMaxP->rows, cuSoftMaxP->cols>>>(cuSoftMaxP->devData, smr.b->devData);
	cudaDeviceSynchronize();

}

/*
	函数功能   ：计算回归层的代价
	softMaxP   : 回归层激活值
	groundTruth: 标签的矩阵
	cost       : 代价结果写入这里
	y          : b标签
	rows       : 数据的行，
	cols       : 数据的列
	batch      : 样本的个数
	线程分配   ：由于涉及到求和操作，而且数据量不是特别大，所以开启
				<<<dim3(1),dim3(256)>>>这样可以避免要去同步不同block之间的数据
*/
__global__ void g_getCost_1(double* softMaxP,
	double* groundTruth, double* cost, double*y, int rows, int cols, int batch)
{
	extern __shared__ double _sum[];
	//grounTruth 清零
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
	//grounTruth赋值
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

	//求和
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
	函数功能：softMax层的权重代价
	cost    : 代价的输出 
	weight  : 权重
	lambda  : 系数
	rows    : 权重的行数
	cols    : 权重的列数
	线程分配：<<<dim3(1),dim3(256)>>>
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

 __device__ double atomicAdd(double* address, double val)
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
	函数功能：卷积层的代价函数的计算
	cost    : 代价的输出 
	weight  : 权重
	lambda  : 系数
	rows    : 权重的行数
	cols    : 权重的列数
	线程分配：<<<dim3(卷积核的个数),dim3(256)>>>
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
	double*y,
	std::vector<cuCvl> &CLayers, 
	std::vector<cuNtw> &hLayers,
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
		g_getCost_3<<<dim3(Config::instance()->getConv()[cl]->m_amount), dim3(32), sizeof(double) * 32>>>(smr.cost->devData, CLayers[cl].d_w, lambda,
			Config::instance()->getConv()[cl]->m_kernelSize, Config::instance()->getConv()[cl]->m_kernelSize);
		cudaDeviceSynchronize();
	}
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
	线程安排:<<<dim3(1),dim3(256)>>>
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
	函数功能：求解bgrad
	线程分配<<<dim3(特征数),dim3(样本数)>>>
	每个block计算一个特征,采用二分的方法
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
	函数功能：求解bgrad
	线程分配<<<dim3(特征数),dim3(样本数)>>>
	每个block计算一个特征,采用二分的方法
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
}

__global__ void g_dnonLinearity(double* delta, double*acti, int len, int NONLIN)
{
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = threadIdx.x + i;
		if(id < len)
		{	
			delta[id] *= d_dnonLinearity(acti[id], NONLIN);
		}
	}
}

__global__ void g_getHiddlenWgrad(double* wgrad, double* dropM, int len, int batch)
{
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			wgrad[id] /= batch;
			wgrad[id] *= dropM[id];
		}
	}
}

void getHiddenDelta(
	std::vector<cuNtw> &hLayers,
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
 			g_dnonLinearity<<<dim3(1), dim3(256)>>>(cuHiddenDelta[hl]->devData,
				cuHiddenActi[hl]->devData, cuHiddenDelta[hl]->getLen(), Config::instance()->getNonLinearity());
 			cudaDeviceSynchronize();
		}
		else
		{
			matrixMul(cuHiddenDelta[hl + 1], hLayers[hl + 1].afterDropW,
				cuHiddenDelta[hl], handle);
			g_dnonLinearity<<<dim3(1), dim3(256)>>>(
				cuHiddenDelta[hl]->devData, 
				cuHiddenActi[hl]->devData,
				cuHiddenDelta[hl]->getLen(), Config::instance()->getNonLinearity());
			cudaDeviceSynchronize();
		}
	}

	for(int hl = Config::instance()->getFC().size() - 1; hl >= 0; hl--)
	{
		if(hl != 0)
		{
			matrixMulTA(cuHiddenDelta[hl],
				cuHiddenActi[hl - 1],
				hLayers[hl].Wgrad, handle);
			g_getHiddlenWgrad<<<dim3(1), dim3(256)>>>(hLayers[hl].Wgrad->devData, hLayers[hl].dropW->devData,
				hLayers[hl].Wgrad->getLen(), batch);
			cudaDeviceSynchronize();
		
		}
		else
		{
			matrixMulTA(cuHiddenDelta[hl],
				cuPool[cuPool.size() - 1],
				hLayers[hl].Wgrad, handle);
			g_getHiddlenWgrad<<<dim3(1), dim3(256)>>>(hLayers[hl].Wgrad->devData, hLayers[hl].dropW->devData, 
				hLayers[hl].Wgrad->getLen(), batch);
			cudaDeviceSynchronize();
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
	}
}

/*
	函数功能：反向传导中的unpooling操作，主要根据
			  记录的pointx和pointy数据，对delta向池化层的
			  上一层传导
	线程分配：<<<dim3(convLen / (convSize * convSize)) / batch, batch
				dim3(poolsize, poolSize)>>>

*/
__global__ void g_unPooling(int* pointX, int* pointY,
	double* prePool, double* curConv,
	int poolSize, int poolDim, int convSize, int len)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id >= len)
		return;

	int convId = id / poolSize / poolSize;
	int idx    = id % (poolSize * poolSize);
	//blockIdx.x + blockIdx.y * gridDim.x;
	int poolSkip = poolSize * poolSize * convId;
	int*       x = pointX + poolSkip;
	int*       y = pointY + poolSkip;
	double* pool = prePool+ poolSkip;
	double* conv = curConv+ convSize * convSize * convId;
	int    curX = x   [idx];
	int    curY = y   [idx];
	double curP = pool[idx];
	conv[curX * convSize + curY] = curP;
}

/*
	函数功能：将第二层的convdelta反向推导到第一层的pooldelta，
			  这里需要经过第二层的核函数，需要将核函数旋转180度。
	线程分配：<<<dim3(batch, k1 * k2),dim3(nxtSize, nxtSize)>>>
			  由于涉及的加法(reduce操作）,但是每个线程只负责累加k1个元素，
			  所以直接用原子操作就可以达到目的（累加元素过多，
			  则需要采用二分的方法来实现）

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
	int     len)  
{
	int id = blockIdx.y * blockDim.y + threadIdx.y;
	if(id >= len)return;

	int curSize          = _convOutputSize;
	int curAddBorderSize = _poolOutputSize;
	int wSize            = _kernelSize;
	int nxtSize          = _poolOutputSize;

// 	g_dConvAdd<<<dim3(batch, cuKernelAmount[cl - 1] * Config::instance()->getConv()[cl]->m_amount),
// 		dim3(cuPoolOutputSize[cl - 1] * cuPoolOutputSize[cl - 1])>>>

	int k1 = id / _kernelAmount2;
	int k2 = id % _kernelAmount2;
	int s  = blockIdx.x;
	int i  = threadIdx.x / _poolOutputSize;
	int j  = threadIdx.x % _poolOutputSize;

	int curSize2 = curSize * curSize;
	int nxtSize2 = nxtSize * nxtSize;
	double* curDelta = _convDelta 
		+ curSize2 * s * _kernelScan2
		+ curSize2 * k1* _kernelAmount2
		+ curSize2 * k2;
	double* nxtDelta = _poolDelta 
		+ nxtSize2 * s * _kernelScan1
		+ nxtSize2 * k1;
	double* addBorder= _addBorder	
		+ nxtSize2 * s * _kernelScan2
		+ nxtSize2 * k1* _kernelAmount2
		+ nxtSize2 * k2;
	double*        w = _w[k2];

	if(i < curSize && j < curSize)
	{
		double val = curDelta[i * curSize + j];
		int x = i + (wSize >> 1);
		int y = j + (wSize >> 1);
		addBorder[x * curAddBorderSize + y] = val;
	}

	__syncthreads();
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
	atomicAdd(nxtDelta + i * nxtSize + j, val);
}

/*
	
	函数功能：求解卷积层的wgrad的第一个函数，由于涉及计算量比较大的reduce操作，所以不建议直接用actom操作，
			  而是
	线程分配：dim3<<<dim3(batch, kernelAmount2), dim3(nxtSize * nxtSize, kernelAmount1)>>>
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
	int len)
{
	int id = blockIdx.y * blockDim.y + threadIdx.y;
	if(id >= len)
		return;
// 	g_conv<<<dim3(batch,  Config::instance()->getConv()[cl]->m_amount),
// 		dim3(Config::instance()->getConv()[cl]->m_kernelSize * Config::instance()->getConv()[cl]->m_kernelSize, cuKernelAmount[cl - 1])>>>(
	int curSize = poolOutputSize;
	int wSize   = convOutputSize;
	int nxtSize = kernelSize;

	int s = blockIdx.x;

	int k2= id % kernelAmount2;
	int k1= id / kernelAmount2;

	int i = threadIdx.x / nxtSize;
	int j = threadIdx.x % nxtSize;

	int curSize2 = curSize * curSize;
	int wSize2   = wSize   * wSize;
	int nxtSize2 = nxtSize * nxtSize;
	double* cur   = pool
		+ curSize2 * s * kernelScan1
		+ curSize2 * k1;

	double* w     = convDelta
		+ wSize2 * s * kernelScan2
		+ wSize2 * k1* kernelAmount2
		+ wSize2 * k2;

	double* nxt   = WgradTmp
		+ nxtSize2 * s * kernelScan2
		+ nxtSize2 * k1* kernelAmount2
		+ nxtSize2 * k2;

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
	nxt[i * nxtSize + j] = val;
}

/*
	线程分配：<<<dim3(k2, kernelSize*kernelSize), dim3(256)>>>
*/
__global__ void g_convAdd(double* WgradTmp, double** Wgrad,
	double** w,
	int _len, 
	int kernelScan1,
	int kernelScan2,
	int kernelAmount1,
	int kernelAmount2,
	int kernelSize,
	int batch,
	double lambda)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	int kid= blockIdx.y;
	int tid= threadIdx.x;

	_sum[threadIdx.x] = 0;
	__syncthreads();
	int kernelSize2 = kernelSize * kernelSize;
	int  tlen = _len / kernelSize2 / kernelAmount2;
	for(int i = 0; i <  tlen; i += blockDim.x)
	{
		int idx = i + threadIdx.x;
		if(idx < tlen)
		{
			int s = idx / kernelScan1;
			int k1= idx % kernelScan1;

			int id = 
				kernelSize2 * s * kernelScan2
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
		if(tid < (len >> 1))
		{
			_sum[tid] += _sum[tid + skip];
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();

	if(tid == 0)
	{
		Wgrad[k2][kid] = _sum[0] / batch + w[k2][kid] * lambda;
	}
}

/*
	线程分配：<<<dim3(kernelAmount2), dim3(256)>>>
*/
__global__ void g_getCLayerBgrad(double* delta,
	double** bgrad,
	int deltaSize,
	int kernelScan1,
	int kernelScan2,
	int kernelAmount1,
	int kernelAmount2,
	int batch,
	int _len)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	_sum[threadIdx.x] = 0.0;
	__syncthreads();
	int tlen = _len / kernelAmount2;
	int deltaSize2 = deltaSize * deltaSize;
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
				deltaSize2 * s * kernelScan2
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
		bgrad[k2][0] = _sum[0] / batch;
	}
}

/*
	
	函数功能：求解卷积层的wgrad的第一个函数，由于涉及计算量比较大的reduce操作，
			所以不建议直接用actom操作，而是采用二分来求和
	线程分配：dim3<<<dim3(batch), dim3(nxtSize * nxtSize, kernelAmount1)>>>
*/
__global__ void g_conv_1(double** sArray,
	double* convDelta,
	double* WgradTmp,
	int imgSize,
	int convOutputSize,
	int kernelScan2,
	int kernelAmount1,
	int kernelSize)
{
	int curSize = imgSize;
	int wSize   = convOutputSize;
	int nxtSize = kernelSize;

	int s = blockIdx.x;
	int k2= threadIdx.y;
	int i = threadIdx.x / nxtSize;
	int j = threadIdx.x % nxtSize;

	int wSize2   = wSize * wSize;
	int nxtSize2 = nxtSize * nxtSize;
	double* cur   = sArray[s];

	double* w     = convDelta
		+ wSize2 * s * kernelScan2
		+ wSize2 * k2;

	double* nxt   = WgradTmp
		+ nxtSize2 * s * kernelScan2
		+ nxtSize2 * k2;

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
	nxt[i * nxtSize + j] = val;
}


/*
	线程分配：<<<dim3(k1, kernelSize*kernelSize), dim3(256)>>>
*/
__global__ void g_convAdd_1(double* WgradTmp, double** Wgrad,
	double** w,
	int _len, 
	int kernelScan2,
	int kernelAmount2,
	int kernelSize,
	int batch,
	double lambda)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	int kid= blockIdx.y;
	int tid= threadIdx.x;

	_sum[threadIdx.x] = 0;
	__syncthreads();
	int kernelSize2 = kernelSize * kernelSize;
	int tlen = _len / kernelSize2 / kernelAmount2;
	for(int i = 0; i < tlen; i += blockDim.x)
	{
		int s = (i + threadIdx.x);
		if(s < tlen)
		{
			int id = 
				kernelSize2 * s * kernelScan2
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
		Wgrad[k2][kid] = _sum[0] / batch + w[k2][kid] * lambda;
	}
}

/*
	线程分配：<<<dim3(kernelAmount2), dim3(256)>>>
*/
__global__ void g_getCLayerBgrad_1(double* delta,
	double** bgrad,
	int deltaSize,
	int kernelScan2,
	int kernelAmount2,
	int batch,
	int _len)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	_sum[threadIdx.x] = 0.0;
	__syncthreads();
	int tlen = _len / kernelAmount2;
	int deltaSize2 = deltaSize * deltaSize;
	for(int i = 0; i < tlen; i += blockDim.x)
	{
		int idx = i + threadIdx.x;
		if(idx < tlen)
		{
			int s  = idx / (deltaSize2);//s
			int t2 = idx % (deltaSize2);//x,y

			int id = 
				deltaSize2 * s * kernelScan2
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
		bgrad[k2][0] = _sum[0] / batch;
	}
}


void dConvAndUnpooling(double**x, 
	std::vector<cuCvl> &CLayers,
	std::vector<cuNtw> &hLayers,
	double lambda,
	int batch, int ImgSize, int nclasses, cublasHandle_t handle)
{
	matrixMul(cuHiddenDelta[0], hLayers[0].afterDropW, 
		cuPoolDelta[cuPoolDelta.size() - 1], handle);

	for(int cl = CLayers.size() - 1; cl >= 0; cl--)
	{
		cuConvDelta[cl]->gpuClear();
		if(cuPoolOutputSize[cl] * cuPoolOutputSize[cl] > MAX_THREADS)
		{
			printf("dConvAndUnpooling g_unPooling cuPoolOutputSize[cl] * cuPoolOutputSize[cl] > MAX_THREADS\n");
			exit(0);
		}
		int len = cuConvDelta[cl]->getLen() / (cuConvOutputSize[cl] * cuConvOutputSize[cl]) * cuPoolOutputSize[cl] * cuPoolOutputSize[cl];
		g_unPooling<<<dim3((len + 511) / 512),
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
		g_dnonLinearity<<<dim3(1),dim3(256)>>>(cuConvDelta[cl]->devData,
			cuConv[cl]->devData, cuConvDelta[cl]->getLen(), Config::instance()->getNonLinearity());
		cudaDeviceSynchronize();

		if(cl > 0)
		{
			cuPoolDelta[cl - 1]->gpuClear();
			cuPoolDeltaAndBorder[cl - 1]->gpuClear();
			//cppPoolDelta
			if(cuPoolOutputSize[cl - 1] * cuPoolOutputSize[cl - 1] > MAX_THREADS)
			{
				printf("g_dConvAdd threads > MAX_THREADS\n");
				exit(0);
			}
			int len = cuKernelScan[cl - 1] * Config::instance()->getConv()[cl]->m_amount;
			int threadidx = cuPoolOutputSize[cl - 1] * cuPoolOutputSize[cl - 1];
			int threadidy = min(512, len + threadidx - 1) / threadidx;//总线程数不能超过1024
			int blockidy  = (len + threadidy -1) / threadidy;

			if(threadidx * threadidy > MAX_THREADS)
			{
				printf("g_dConvAdd > MAX_THREADS\n");
				exit(0);
			}

			g_dConvAdd<<<dim3(batch, blockidy),
				dim3(threadidx, threadidy)>>>
			(
				cuConvDelta[cl]->devData,
				cuPoolDeltaAndBorder[cl - 1]->devData,
				CLayers[cl].d_w,
				cuPoolDelta[cl - 1]->devData,
				cuConvOutputSize[cl],
				cuPoolOutputSize[cl - 1],
				cuKernelScan[cl - 1],
				cuKernelScan[cl],
				Config::instance()->getConv()[cl - 1]->m_amount,
				Config::instance()->getConv()[cl]->m_amount,
				Config::instance()->getConv()[cl]->m_kernelSize,
				len
			);
			cudaDeviceSynchronize();

			len = cuKernelScan[cl - 1] * Config::instance()->getConv()[cl]->m_amount;
			threadidx = Config::instance()->getConv()[cl]->m_kernelSize * Config::instance()->getConv()[cl]->m_kernelSize;
			threadidy = min(512, len + threadidx - 1) / threadidx;//总线程数不能超过1024
			blockidy  = (len + threadidy -1) / threadidy;

			if(threadidx * threadidy> MAX_THREADS)
			{
				printf("g_conv > MAX_THREADS\n");
				exit(0);
			}
			g_conv<<<dim3(batch,  blockidy),
				dim3(threadidx, threadidy)>>>(
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
				len);
			cudaDeviceSynchronize();

			g_convAdd<<<dim3(Config::instance()->getConv()[cl]->m_amount, Config::instance()->getConv()[cl]->m_kernelSize * Config::instance()->getConv()[cl]->m_kernelSize),
				dim3(256),
				sizeof(double) * 256>>>(
				cuConvLayerWgradTmp[cl]->devData,
				CLayers[cl].d_wgrad,
				CLayers[cl].d_w,
				cuConvLayerWgradTmp[cl]->getLen(),
				cuKernelScan[cl - 1],
				cuKernelScan[cl],
				Config::instance()->getConv()[cl - 1]->m_amount,
				Config::instance()->getConv()[cl]->m_amount,
				Config::instance()->getConv()[cl]->m_kernelSize,
				batch,
				lambda);
			cudaDeviceSynchronize();

			g_getCLayerBgrad<<<dim3(Config::instance()->getConv()[cl]->m_amount), 
				dim3(256),
				sizeof(double) * 256>>>(cuConvDelta[cl]->devData,
				CLayers[cl].d_bgrad,
				cuConvOutputSize[cl],
				cuKernelScan[cl - 1],
				cuKernelScan[cl],
				Config::instance()->getConv()[cl - 1]->m_amount,
				Config::instance()->getConv()[cl]->m_amount,
				batch,
				cuConvDelta[cl]->getLen());
			cudaDeviceSynchronize();
		}
		else
		{
			//线程分配：dim3<<<dim3(batch), dim3(nxtSize * nxtSize, kernelSize1)>>>
			if(Config::instance()->getConv()[cl]->m_kernelSize * Config::instance()->getConv()[cl]->m_kernelSize * 
				Config::instance()->getConv()[cl]->m_amount > MAX_THREADS)
			{
				printf("g_conv_1 threads > MAX_THREADS\n");
				exit(0);
			}
			g_conv_1<<<dim3(batch),dim3(Config::instance()->getConv()[cl]->m_kernelSize * Config::instance()->getConv()[cl]->m_kernelSize,
				Config::instance()->getConv()[cl]->m_amount)>>>
				(x,cuConvDelta[cl]->devData,
				cuConvLayerWgradTmp[cl]->devData,
				ImgSize,
				cuConvOutputSize[cl],
				cuKernelScan[cl],
				Config::instance()->getConv()[cl]->m_amount,
				Config::instance()->getConv()[cl]->m_kernelSize);
			cudaDeviceSynchronize();

			if(Config::instance()->getConv()[cl]->m_kernelSize * Config::instance()->getConv()[cl]->m_kernelSize > MAX_THREADS)
			{
				printf("g_convAdd_1 > MAX_THREADS\n");
				exit(0);
			}
			g_convAdd_1<<<dim3(Config::instance()->getConv()[cl]->m_amount, Config::instance()->getConv()[cl]->m_kernelSize * Config::instance()->getConv()[cl]->m_kernelSize),dim3(256),
				sizeof(double) * 256>>>(cuConvLayerWgradTmp[cl]->devData,
				CLayers[cl].d_wgrad,CLayers[cl].d_w,cuConvLayerWgradTmp[cl]->getLen(),cuKernelScan[cl],
				Config::instance()->getConv()[cl]->m_amount, Config::instance()->getConv()[cl]->m_kernelSize,batch, lambda);
			cudaDeviceSynchronize();

			g_getCLayerBgrad_1<<<dim3(Config::instance()->getConv()[cl]->m_amount),dim3(256), sizeof(double) * 256>>>
				(cuConvDelta[cl]->devData,CLayers[cl].d_bgrad, cuConvOutputSize[cl], cuKernelScan[cl],
				Config::instance()->getConv()[cl]->m_amount, batch, cuConvDelta[cl]->getLen());
			cudaDeviceSynchronize();
		}
	}
}


__global__ void g_vecAdd(double*v_w, double*wgrad,double* w,
	double* v_b, double* bgrad, double* b, 
	int lenw, int lenb,
	double momentum, double lrate)
{
	for(int i = 0; i < lenw; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < lenw)
		{
			v_w[id] = v_w[id] * momentum + wgrad[id] * lrate;
			w[id] -= v_w[id];
		}
	}

	for(int i = 0; i < lenb; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < lenb)
		{
			v_b[id] = v_b[id] * momentum + bgrad[id] * lrate;
			b[id] -= v_b[id];
		}
	}
}

void updataWB(std::vector<cuCvl> &CLayers, 
	std::vector<cuNtw> &hLayers,
	cuSMR &smr,
	double lrate,
	double momentum,
	int batch)
{
	g_vecAdd<<<dim3(1),dim3(256)>>>(cu_v_smr_w->devData, smr.Wgrad->devData, smr.Weight->devData,
		cu_v_smr_b->devData, smr.bgrad->devData, smr.b->devData, 
		smr.Wgrad->getLen(),smr.bgrad->getLen(), momentum, lrate);

	for(int i = 0; i < hLayers.size(); i++)
	{
		g_vecAdd<<<dim3(1),dim3(512)>>>(cu_v_hl_w[i]->devData, hLayers[i].Wgrad->devData, hLayers[i].W->devData,
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
}

void getNetworkCost(double** x, 
	double* y , 
	std::vector<cuCvl> &CLayers, 
	std::vector<cuNtw> &hLayers,
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
__global__ void getCorrect(double* softMaxP, double*correct, double* y, int cols)
{
	int id = threadIdx.x;
	if(id == 0)correct[0] = 0;
	double* p = softMaxP + id * cols;
	int r = 0;
	double maxele = log(p[0]);
	for(int i = 0; i < cols; i++)
	{
		double val = log(p[i]);
		if(maxele < val)
		{
			maxele = val;
			r = i;
		}
	}

	if(fabs(r - y[id]) <= 0.000001f)
	{
		atomicAdd(correct, 1);
	}
}

int resultProdict(double** testX, double*testY,
	std::vector<cuCvl> &CLayers, 
	std::vector<cuNtw> &hLayers, 
	cuSMR &smr, double lambda, int batch, int ImgSize, int nclasses, cublasHandle_t handle)
{
	convAndPooling(testX, CLayers, batch, ImgSize);
	getHiddenLayerActi(hLayers, handle);
	getSoftMaxP(smr, handle);
	getCorrect<<<dim3(1), batch>>>(
		cuSoftMaxP->devData, 
		cuCorrect->devData, 
		testY, 
		cuSoftMaxP->cols);
	cudaDeviceSynchronize();
	cuCorrect->toCpu();
	return cuCorrect->hostData[0];
}

__global__ void g_resultPredictAdd(double* correctCount, double* predict, int cols, int len)
{
	for(int i = 0; i < len ; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			double* count = correctCount + id * cols;
			count[(int)predict[id]] += 1.0;
		}
	}
}

__global__ void g_resultCountProdict(double* correcttCount, double* y, double* correct, int cols, int len)
{
	if(threadIdx.x == 0)correct[0] = 0;
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			double* count = correcttCount + id * cols;
			double r = 0;
			double maxele =  count[0];

			for(int j = 0; j < cols; j++)
			{
				if(maxele < count[j])
				{
					maxele = count[j];
					r = j;
				}
			}

			if(fabs(r - y[id]) <= 0.000001f)
			{
				atomicAdd(correct, 1);
			}
		}
		
	}

}

void gradientChecking(std::vector<cuCvl> &CLayers, std::vector<cuNtw> &hLayers, cuSMR &smr, double**x, 
	double*y, double lambda, int batch, int ImgSize, int nclasses, cublasHandle_t handle)
{
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
				CLayers[a].layer[b].Wgrad->cols);
			
			for(int i = 0; i < CLayers[a].layer[b].W->rows; i++)
			{
				for(int j = 0; j < CLayers[a].layer[b].W->cols; j++)
				{
					double memo = CLayers[a].layer[b].W->get(i, j);
					CLayers[a].layer[b].W->set(i, j, memo + epsilon);
					CLayers[a].layer[b].W->toGpu();
					getNetworkCost(x, y, CLayers, hLayers, smr, lambda, batch, ImgSize, nclasses, handle);
					smr.cost->toCpu();
					double value1 = smr.cost->get(0,0);
					CLayers[a].layer[b].W->set(i, j, memo - epsilon);
					CLayers[a].layer[b].W->toGpu();
					getNetworkCost(x, y, CLayers, hLayers, smr, lambda, batch, ImgSize, nclasses, handle);
					smr.cost->toCpu();
					double value2 = smr.cost->get(0, 0);
					double tp = (value1 - value2) / (2 * epsilon);
					if(int(10000 * grad->get(i,j)) != int(10000 * tp))
						std::cout<<i<<", "<<j<<", "<<tp<<", "<<grad->get(i,j)<<", "
							<<tp / grad->get(i,j) <<std::endl;
					CLayers[a].layer[b].W->set(i, j, memo);
					CLayers[a].layer[b].W->toGpu();
				}
			}

			delete grad;
		}
	}
}

void cuTrainNetwork(cuMatrixVector<double>&x, 
	cuMatrix<double>*y , 
	std::vector<cuCvl> &CLayers,
	std::vector<cuNtw> &HiddenLayers, 
	cuSMR &smr,
	double lambda, 
	cuMatrixVector<double>&testX,
	cuMatrix<double>* testY, 
	int nsamples,
	int batch,
	int ImgSize, 
	int nclasses,
	cublasHandle_t handle)
{
// 	gradientChecking(CLayers, HiddenLayers, smr, cu_D_trainX, y->devData, 
// 		lambda, batch, handle);

	cuCorrectCount->gpuClear();
	int correct = 0;
	for(int p = 0; p < testX.size() / batch; p++)
	{
		int tstart = p * batch;
		correct += resultProdict(testX.m_devPoint + tstart,  testY->devData + tstart, 
			CLayers, HiddenLayers, smr, lambda, batch, ImgSize, nclasses, handle);	
	}

	if(correct >  cuCurCorrect)
	{
		cuCurCorrect = correct;
		cuSaveConvNet(CLayers, HiddenLayers, smr, ImgSize, nclasses, nsamples);
	}
	printf(" = %d\n", correct);

	int epochs = 20000;
 	double nlrate[] =    {0.05, 0.04, 0.03, 0.02, 0.01, 0.008, 0.006, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 
		0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125, 0.000390625, 0.0001953125, 0.00009765625, 0.000048828125, 0.0000244140625};
 	double nMomentum[] = {0.90, 0.91, 0.92, 0.93, 0.94, 0.942, 0.944, 0.95,    0.96,   0.97,  0.995,  0.90,  
		0.92,  0.93,   0.94,    0.95,     0.96,      0.97,       0.98,        0.99,         0.994,         0.995,          0.996};
 	int epoCount[] =     {80,   80,     80,   80,   80,    80,    80,   80,      80,     80,     80,   80,   
		80,    80,     80,      80,       80,        80,         80,          80,           80,            80,             80};

	printf("%d %d %d", sizeof(nlrate), sizeof(nMomentum), sizeof(epoCount));
	while(1)
	{
		double lrate = 0.05;
		double Momentum = 0.9;
		int id = 0;
		for(int epo = 0; epo < epochs; epo++){
			if(id >= sizeof(nlrate) / sizeof(double))
				break;
			lrate = nlrate[id];
			Momentum = nMomentum[id];
	
			double start,end;
			start = clock();

			cuApplyRandom(batch,  time(NULL), 3.4, ImgSize);
			for(int hl = 0; hl < HiddenLayers.size(); hl++)
			{
				dropDelta(HiddenLayers[hl].dropW, Config::instance()->getFC()[hl]->m_dropoutRate);
			}
			for(int k = 0; k < (x.size() + batch - 1) / batch; k++)
			{
				int start = rand() % (x.size() - batch);

				cuApplyDistortion(x.m_devPoint + start, cu_distortion_vector->m_devPoint, batch, ImgSize);
				cuApplyCrop(cu_distortion_vector->m_devPoint, cu_distortion_vector->m_devPoint, batch, ImgSize);

// 				for(int ff = 0; ff < batch; ff++)
// 				{
// 					showImg(x[start + ff], 10);
// 					showImg(cu_distortion_vector->m_vec[ff], 10);
// 					cv::waitKey(0);
// 				}

				getNetworkCost(cu_distortion_vector->m_devPoint,
					y->devData + start,
					CLayers, HiddenLayers,
					smr,
					lambda, batch, ImgSize, nclasses, handle);
				updataWB(CLayers, HiddenLayers, smr, lrate, Momentum, batch);
			}

			smr.cost->toCpu();
			char str[512];
			int correct = 0;
			for(int hl = 0; hl < HiddenLayers.size(); hl++)
			{
				dropDelta(HiddenLayers[hl].dropW, 0.0f);
			}
			for(int p = 0; p < testX.size() / batch; p++)
			{
				int tstart = p * batch;
				correct += resultProdict(testX.m_devPoint + tstart,  testY->devData + tstart, 
					CLayers, HiddenLayers, smr, lambda, batch, ImgSize, nclasses, handle);
			}

			if(correct >  cuCurCorrect)
			{
				cuCurCorrect = correct;
				cuSaveConvNet(CLayers, HiddenLayers, smr, ImgSize, nclasses, nsamples);
			}

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
				(double)(end - start) / CLOCKS_PER_SEC, smr.cost->get(0,0), correct, cuCurCorrect, Momentum, lrate);

			printf("%s\n",str);
			LOG(str,"output");
		}
	}
}

void cuClearCorrectCount()
{
	cuCorrectCount->gpuClear();
}

void __global__ g_resultCopy(double* predict, double* softMax, int nclass)
{
	double* sm = softMax + threadIdx.x * nclass;
	double _max = sm[0];
	int id = 0;
	for(int i = 1; i < nclass; i++)
	{
		if(sm[i] > _max)
		{
			_max = sm[i];
			id = i;
		}
	}
	predict[threadIdx.x] = id;
}

int cuPredictNetwork(cuMatrixVector<double>& x, 
	cuMatrix<double>*y , 
	std::vector<cuCvl> &CLayers,
	std::vector<cuNtw> &HiddenLayers, 
	cuSMR &smr,
	double lambda, 
	cuMatrixVector<double>& testX,
	cuMatrix<double>* testY, 
	cuMatrix<double>* predict,
	int imgDim, 
	int nsamples,
	int batch,
	int ImgSize,
	int nclasses,
	cublasHandle_t handle)
{
	int correct = 0;
	for(int p = 0; p < (x.size() + batch - 1)/ batch; p++)
	{
		int tstart = p * batch;
		if(tstart + batch > x.size())
		{
			tstart = x.size() - batch;
		}
		resultProdict(x.m_devPoint + tstart,  y->devData + tstart,
			CLayers, HiddenLayers, smr, lambda, batch, ImgSize, nclasses, handle);
		g_resultCopy<<<dim3(1),dim3(batch)>>>(predict->devData + tstart, cuSoftMaxP->devData, nclasses);
		cudaDeviceSynchronize();
	}
	predict->toCpu();
	for(int i = 0; i < predict->getLen(); i++)
	{
		if(predict->get(i, 0) != y->hostData[i])
		{
			printf("id=%d predict=%lf correct=%lf\n", i, predict->get(i, 0), y->hostData[i]);
		}
		else 
		{
			correct++;
		}
	}
	return correct;
}

int cuPredictAdd(cuMatrix<double>* predict, cuMatrix<double>* testY, int batch, int ImgSize, int nclasses)
{
	g_resultPredictAdd<<<dim3(1), dim3(batch)>>>(cuCorrectCount->devData, predict->devData, nclasses, testY->getLen());
	cudaDeviceSynchronize();
	g_resultCountProdict<<<dim3(1),dim3(batch)>>>(cuCorrectCount->devData, testY->devData, cuCorrect->devData, nclasses, testY->getLen());
	cudaDeviceSynchronize();
	cuCorrect->toCpu();
	return  cuCorrect->get(0,0);
}

void cuShowInCorrect(cuMatrixVector<double>&testX, cuMatrix<double>* testY, int ImgSize, int nclasses)
{
	cuCorrectCount->toCpu();
	testY->toCpu();
	for(int i = 0; i < testY->getLen(); i++)
	{
		int _max = 0;
		int id = 0;
		for(int j = 0; j <  nclasses; j++)
		{
			if(cuCorrectCount->get(i, j) > _max)
			{
				_max = cuCorrectCount->get(i, j);
				id = j;
			}
		}
		if(id != testY->hostData[i])
		{
			for(int j = 0; j < nclasses; j++)
			{
				printf("%.0lf ", cuCorrectCount->get(i,j));
			}printf("\n");
			printf("predictId = %d correctId = %.0lf\n", id, testY->hostData[i]);
			//cuApplyCrop(cu_D_Distortion, cu_D_Distortion, 1);
			showImg(testX[i], 10);
			cv::waitKey(0);
		}
	}
}