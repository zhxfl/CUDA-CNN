#include "net.cuh"
#include "opencv2/opencv.hpp"
#include "common/cuMatrix.h"
#include <cuda_runtime.h>
#include "common/util.h"
#include <time.h>
#include "dataAugmentation/cuTrasformation.cuh"
#include "common/Config.h"
#include "common/cuMatrixVector.h"
#include <helper_functions.h>
#include <helper_cuda.h>
#include "common/MemoryMonitor.h"
#include "layers/Pooling.h"
#include "common/cuBase.h"
#include "layers/ConvCFM.h"
#include "layers/ConvNCFM.h"
#include "layers/FullConnect.h"


cuMatrixVector<double>* cu_distortion_vector;

/*temp struct for matrix rearrange*/
//cuMatrix<double>* cuPoolToFlActi;

/*full connect layer activity*/
//std::vector<cuMatrix<double>*>cuFullConnectActi;

/*softMax*/
cuMatrix<double>* cuSoftMaxP;
cuMatrix<double>* cuGroundTruth;

/*softMax delta*/
cuMatrix<double>* cuSoftMaxDelta;
//std::vector<cuMatrix<double>*>cuFullConnectDelta;

/*temp struct for matrix rearrange*/
//cuMatrix<double>* cuPoolToFlDelta;

/*momentum*/
cuMatrix<double>* cu_v_smr_w;
cuMatrix<double>* cu_v_smr_b;
//std::vector<cuMatrix<double>*>cu_v_hl_w;
//std::vector<cuMatrix<double>*>cu_v_hl_b;
int cuCurCorrect;
cuMatrix<int>*cuCorrect = NULL;
cuMatrix<int>*cuVote = NULL;

std::vector<Pooling*>poolings;
std::vector<ConvCFM*>convCFM;
std::vector<ConvNCFM*>convNCFM;
std::vector<FullConnect*>fullConnect;

/*batch size images*/
cuMatrixVector<double>batchImg[2];

void getBatchImageWithStreams(cuMatrixVector<double>&x, cuMatrixVector<double>&batchImg, int start, cudaStream_t stream1);
void outputMatrix(cuMatrix<double>* m);

/*
* function: init convolution layer weight as two-dimensional arra
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

/*
* function: convolution kernel weight init
*/

void weightRandomInit(cuConvK &convk, int width){
	double epsilon = 0.1f;
	convk.W = new cuMatrix<double>(width, width, Config::instance()->getChannels());

	for(int c = 0; c < Config::instance()->getChannels(); c++)
	{
		double r1 = 0.5 + 4.0 * (rand()) / RAND_MAX;
		double r2 = 0.5 + 4.0 * (rand()) / RAND_MAX;
		createGaussian(convk.W->hostData + c * convk.W->getArea(), r1,r2,
			width, width, 
			Config::instance()->getChannels(), 
			epsilon * 0.5 + epsilon * rand() / RAND_MAX);
	}
	
// 	for(int c = 0; c < convk.W->channels; c++){
// 		for(int i = 0; i<convk.W->rows; i++){
// 			for(int j=0; j<convk.W->cols; j++){
// 				convk.W->set(i, j, c, 1.0 * rand() / RAND_MAX * 2.0 * epsilon - epsilon,);
// 			}
// 		}
// 	}
	
	
	convk.W->toGpu();
	convk.b = new cuMatrix<double>(1, 1, Config::instance()->getChannels());
	convk.Wgrad = new cuMatrix<double>(width, width, Config::instance()->getChannels());
	convk.bgrad = new cuMatrix<double>(1, 1, Config::instance()->getChannels());
}


/*
* function: full connect layer weight init
*/

void weightRandomInit(cuFll &fll, int inputsize, int FullConnectsize, double dropRate){
	double epsilon = sqrt((double)6) / sqrt((double)(FullConnectsize + inputsize + 1));
	fll.W = new cuMatrix<double>(FullConnectsize, inputsize, 1);
	fll.dropW = new cuMatrix<double>(FullConnectsize, inputsize, 1);
	fll.afterDropW = new cuMatrix<double>(FullConnectsize, inputsize, 1);

	//double r1 = 2.0 + 4.0 * (rand()) / RAND_MAX;
	//double r2 = 2.0 + 4.0 * (rand()) / RAND_MAX;
	//createGaussian(fll.W->hostData, r1, r2, fll.W->rows, fll.W->cols, epsilon * 0.5 + epsilon * rand() / RAND_MAX);
	for(int c=0; c < fll.W->channels; c++){
		for(int i=0; i < fll.W->rows; i++){
			for(int j=0; j< fll.W->cols; j++){
				fll.W->set(i,j, c, 1.0 * rand() / RAND_MAX *  2.0 * epsilon - epsilon);
			}
		}
	}

	fll.W->toGpu();
	fll.b = new cuMatrix<double>(FullConnectsize, 1, 1);
	fll.Wgrad = new cuMatrix<double>(FullConnectsize, inputsize, 1);
	fll.bgrad = new cuMatrix<double>(FullConnectsize, 1, 1);
}

/*
* function: softMax weight init
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

void cuConvNetInitPrarms(
	cuSMR &smr,
	int imgDim,
	int nclasses)
{
	srand(clock());
	/*Init Softmax layer*/
	
}

void saveWeight(cuConvK &convk, FILE*pOut)
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

void saveWeight(cuFll &fll, FILE*pOut){
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

void saveWeight(cuSMR &smr, FILE* pOut){
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

void cuSaveConvNet(
	cuSMR &smr)
{	
	FILE *pOut = fopen("Result/checkPoint.txt", "w");
	/* Init Conv layers*/
	for(int i=0; i < Config::instance()->getConv().size(); i++){
		if(Config::instance()->getCFM() == 0){
			convNCFM[i]->save(pOut);
		}
		else {
			convCFM[i]->save(pOut);
		}

	}

	/* Init FullConnect layers */
	for(int fl = 0; fl < fullConnect.size(); fl++){
		fullConnect[fl]->save(pOut);
	}

	/* Init Softmax layer */
	saveWeight(smr, pOut);
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

void readWeight(cuFll &fll, int inputsize, int FullConnectsize, FILE*pIn, double dropRate){
	double val = 0.0;
	fll.W = new cuMatrix<double>(FullConnectsize, inputsize, 1);
	for(int c = 0; c < fll.W->channels; c++){
		for(int i=0; i<fll.W->rows; i++){
			for(int j=0; j<fll.W->cols; j++){
				fscanf(pIn, "%lf", &val);
				fll.W->set(i,j,c,val);
			}
		}
	}


	fll.dropW = new cuMatrix<double>(FullConnectsize, inputsize, 1);
	fll.afterDropW = new cuMatrix<double>(FullConnectsize, inputsize, 1);
	dropDelta(fll.dropW, dropRate);


	fll.b = new cuMatrix<double>(FullConnectsize, 1, 1);
	for(int i = 0; i < fll.b->rows; i++){
		for(int j = 0; j < fll.b->cols; j++){
			fscanf(pIn, "%lf", &val);
			fll.b->set(i,j,0,val);
		}
	}

	fll.Wgrad = new cuMatrix<double>(FullConnectsize, inputsize, 1);
	fll.bgrad = new cuMatrix<double>(FullConnectsize, 1, 1);
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

void cuFreeConvNet(
	cuSMR &smr)
{
	smr.clear();
}

void cuReadConvNet(
	cuSMR &smr, int imgDim, char* path,
	int nclasses)
{	
	FILE *pIn = fopen(path, "r");

	/*combine feature maps*/
	if (Config::instance()->getCFM() == 0) {
		for(int cl = 0; cl < convNCFM.size(); cl++){
			convNCFM[cl]->initFromCheckpoint(pIn);
		}
	} else {
		for(int cl = 0; cl < convCFM.size(); cl++){
			convCFM[cl]->initFromCheckpoint(pIn);
		}
	}

	for(int fl = 0; fl < fullConnect.size(); fl++){
		fullConnect[fl]->initFromCheckpoint(pIn);
	}

	/*Init Softmax layer*/
	readWeight(smr, nclasses, Config::instance()->getFC()[Config::instance()->getFC().size() - 1]->m_numFullConnectNeurons, pIn);
	fclose(pIn);
};

void cuInitCNNMemory(
	int batch,
	cuMatrixVector<double>& trainX, 
	cuMatrixVector<double>& testX,
	cuSMR& smr,
	int ImgSize,
	int nclasses)
{
	/*Transformation*/
	cu_distortion_vector = new cuMatrixVector<double>();
	for(int i = 0; i < batch; i++){
		cu_distortion_vector->push_back(new cuMatrix<double>(ImgSize, ImgSize, Config::instance()->getChannels()));
	}
	cu_distortion_vector->toGpu();

	FILE *pIn = fopen("Result/checkPoint.txt", "r");

	/* convolution layer and pooling*/
	if(Config::instance()->getCFM()){
		for(int i = 0; i < Config::instance()->getConv().size(); i++){
				if(i == 0){
					convCFM.push_back(new ConvCFM(cu_distortion_vector,
						1,
						Config::instance()->getConv()[i]->m_amount,
						Config::instance()->getConv()[i]->m_kernelSize,
						Config::instance()->getConv()[i]->m_padding,
						ImgSize,
						batch,
						Config::instance()->getConv()[0]->m_weightDecay,
						1,
						Config::instance()->getNonLinearity()));

				
					convCFM[i]->initRandom();

					poolings.push_back(new Pooling(convCFM[i]->getOutputs(),
						Config::instance()->getPooling()[i]->m_size,
						Config::instance()->getPooling()[i]->m_skip,
						convCFM[i]->getOutputDim(), 
						convCFM[i]->getOutputAmount(),
						batch));

					poolings[i]->setPreDelta(convCFM[i]->getCurDelta());
				}
				else {
					convCFM.push_back(new ConvCFM(poolings[i - 1]->getOutputs(),
						poolings[i - 1]->getOutputAmount(),
						Config::instance()->getConv()[i]->m_amount,
						Config::instance()->getConv()[i]->m_kernelSize,
						Config::instance()->getConv()[i]->m_padding,
						poolings[i - 1]->getOutputDim(),
						batch,
						Config::instance()->getConv()[i]->m_weightDecay,
						Config::instance()->getCFM(),
						Config::instance()->getNonLinearity()));

					
						convCFM[i]->initRandom();
					

					convCFM[i]->setPreDelta(poolings[i - 1]->getCurDelta());

					poolings.push_back(new Pooling(convCFM[i]->getOutputs(),
						Config::instance()->getPooling()[i]->m_size,
						Config::instance()->getPooling()[i]->m_skip,
						convCFM[i]->getOutputDim(), 
						convCFM[i]->getOutputAmount(),
						batch));

					poolings[i]->setPreDelta(convCFM[i]->getCurDelta());
				}
		}
	}
	else{
		for(int i = 0; i < Config::instance()->getConv().size(); i++){
			if(i == 0){
				convNCFM.push_back(new ConvNCFM(cu_distortion_vector,
					1,
					Config::instance()->getConv()[i]->m_amount,
					Config::instance()->getConv()[i]->m_kernelSize,
					Config::instance()->getConv()[i]->m_padding,
					ImgSize,
					batch,
					Config::instance()->getConv()[0]->m_weightDecay,
					Config::instance()->getNonLinearity()));

				
					convNCFM[i]->initRandom();
				

				poolings.push_back(new Pooling(convNCFM[i]->getOutputs(),
					Config::instance()->getPooling()[i]->m_size,
					Config::instance()->getPooling()[i]->m_skip,
					convNCFM[i]->getOutputDim(), 
					convNCFM[i]->getOutputAmount(),
					batch));

				poolings[i]->setPreDelta(convNCFM[i]->getCurDelta());
			}
			else {
				convNCFM.push_back(new ConvNCFM(poolings[i - 1]->getOutputs(),
					poolings[i - 1]->getOutputAmount(),
					Config::instance()->getConv()[i]->m_amount,
					Config::instance()->getConv()[i]->m_kernelSize,
					Config::instance()->getConv()[i]->m_padding,
					poolings[i - 1]->getOutputDim(),
					batch,
					Config::instance()->getConv()[i]->m_weightDecay,
					Config::instance()->getNonLinearity()));

				
					convNCFM[i]->initRandom();
	

				convNCFM[i]->setPreDelta(poolings[i - 1]->getCurDelta());

				poolings.push_back(new Pooling(convNCFM[i]->getOutputs(),
					Config::instance()->getPooling()[i]->m_size,
					Config::instance()->getPooling()[i]->m_skip,
					convNCFM[i]->getOutputDim(), 
					convNCFM[i]->getOutputAmount(),
					batch));

				poolings[i]->setPreDelta(convNCFM[i]->getCurDelta());
			}
		}
	}


	for(int i = 0; i < Config::instance()->getFC().size(); i++){
		if(i == 0){
			fullConnect.push_back(new FullConnect(poolings[poolings.size() - 1]->getOutputs(),
				batch,
				Config::instance()->getFC()[i]->m_weightDecay,
				Config::instance()->getFC()[i]->m_numFullConnectNeurons,
				Config::instance()->getFC()[i]->m_dropoutRate,
				Config::instance()->getNonLinearity()));

			
				fullConnect[i]->initRandom();


			fullConnect[i]->setPreDelta(poolings[poolings.size() - 1]->getCurDelta());
		}else{
			fullConnect.push_back(new FullConnect(fullConnect[i - 1]->getOutputs(),
				batch,
				Config::instance()->getFC()[i]->m_weightDecay,
				Config::instance()->getFC()[i]->m_numFullConnectNeurons,
				Config::instance()->getFC()[i]->m_dropoutRate,
				Config::instance()->getNonLinearity()));
			
				fullConnect[i]->initRandom();

			fullConnect[i]->setPreDelta(fullConnect[i - 1]->getCurDelta());
		}
	}


	weightRandomInit(smr, nclasses, Config::instance()->getFC()[Config::instance()->getFC().size() - 1]->m_numFullConnectNeurons);

	/*softMax*/
	cuSoftMaxP    = new cuMatrix<double>(batch, nclasses, 1);
	cuGroundTruth = new cuMatrix<double>(batch, nclasses, 1);

	/*softMax delta*/
	cuSoftMaxDelta = new cuMatrix<double>(batch, nclasses, 1);

	/*
	* translate the cuPoolToFlDelta from cuMatrix(batch, size * channels, 1)
	* to cuPoolDelta  cuMatrix(batch, size, channels)
	*/


	/*Momentum*/
	cu_v_smr_w = new cuMatrix<double>(smr.Weight->rows, smr.Weight->cols, smr.Weight->channels);
	cu_v_smr_b = new cuMatrix<double>(smr.b->rows, smr.b->cols, smr.b->channels);

	/*correct and cuVote*/
	if(cuCorrect == NULL)
	{
		cuCorrect = new cuMatrix<int>(1,1,1);
		cuVote    = new cuMatrix<int>(testX.size(), Config::instance()->getSoftMax()[0]->m_numClasses, 1);
	}

	/*double buffer for batch images*/
	int crop = Config::instance()->getCrop();
	for(int i = 0; i < 2; i ++){
		for(int j = 0; j < batch; j++){
			batchImg[i].push_back(new cuMatrix<double>(ImgSize + crop, ImgSize + crop, Config::instance()->getChannels()));
		}
		batchImg[i].toGpu();
	}

	MemoryMonitor::instance()->printCpuMemory();
	MemoryMonitor::instance()->printGpuMemory();

	
}

void cuFreeCNNMemory(
	int batch,
	cuMatrixVector<double>&trainX, 
	cuMatrixVector<double>&testX,
	cuSMR &smr)
{
	delete cuSoftMaxP;
	delete cuGroundTruth;
	delete cuSoftMaxDelta;


	delete cu_v_smr_w;
	delete cu_v_smr_b;

	delete cu_distortion_vector;
}

/*
*blocks : dim3(batch, cuKernelScan[0], Config::instance()->getChannels()),
*threads: dim3(min(convOutputSize * convOutputSize, 512));
*/
__global__ void g_conv_1(
	double** arrayS,
	double** arrayW,
	double** arrayB,
	double* conv,
	int inputSize,
	int kernelSize,
	int padding,
	int convSize,
	int convArea,
	int batch,
	int k1Amount,
	int NONLIN)
{
	int sp = blockIdx.x;
	int k  = blockIdx.y;
	int c  = blockIdx.z;

	int convSize2  = convSize * convSize;
	int inputSize2 = inputSize* inputSize;
	int kernelSize2= kernelSize * kernelSize;

	int convSkip  = convArea * c + (sp * k1Amount + k) * convSize2;

	double* curInput = arrayS[sp] + c * inputSize2;
	double* w        = arrayW[k]  + c * kernelSize2;
	double  b        = arrayB[k][c];

	double* curConv  = conv   + convSkip;

	/*convolution*/
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
					int xx = x + i - padding;
					int yy = y + j - padding;
					if(xx >= 0 && xx < inputSize && yy >= 0 && yy < inputSize)
						val += curInput[xx * inputSize + yy] * w[i * kernelSize + j];
				}
			}
			curConv[idx] = d_nonLinearity(val + b, NONLIN);
		}
	}
}

/*
*blocks : dim3(batch, cuKernelScan[0], Config::instance()->getChannels()),
*threads: dim3(min(convOutputSize * convOutputSize, 512));
*/
__global__ void g_pooling(
	double* conv,
	double* pool,
	int* pointX,
	int* pointY,
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

	int convSkip  = convArea * c + (sp * k1Amount + k) * convSize2;
	int poolSkip  = poolArea * c + (sp * k1Amount + k) * poolSize2;

	double* curConv  = conv   + convSkip;
	double* curPool  = pool   + poolSkip;
	int* px          = pointX + poolSkip;
	int* py          = pointY + poolSkip;

	/*pooling*/
	for(int tidx = 0; tidx < poolSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < poolSize2)
		{
			int x = idx / poolSize;
			int y = idx % poolSize;

			int curX = x * poolingDim;
			int curY = y * poolingDim;

			cuAssert(curX < convSize && curY < convSize);
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
* blocks : dim3(batch, cuKernelScan[i], Config::instance()->getChannels()),
* threads: dim3(threadidx)
*/

__global__ void g_conv_2(
	double* pool1,
	double** arrayW,
	double** arrayB,
	double* conv2,
	int pool1Size,
	int kernelSize,
	int padding,
	int conv2Size,
	int k1Scan,
	int k2Scan,
	int k1Amount,
	int k2Amount,
	int pool1Area,
	int conv2Area,
	int NONLIN)
{
	int sp = blockIdx.x;
	int c  = blockIdx.z;
	int k2 = blockIdx.y % k2Amount;
	int k1 = blockIdx.y / k2Amount;

	double* w   = arrayW[k2] + kernelSize * kernelSize * c;
	double  b   = arrayB[k2][c];

	int pool1Size2 = pool1Size * pool1Size;
	int conv2Size2 = conv2Size * conv2Size;

	int skip1 = sp * k1Scan + k1;
	int skip2 = sp * k2Scan + k1 * k2Amount + k2;

	double* pl1 = pool1
		+ pool1Area * c
		+ skip1 * pool1Size2;

	double* cv2 = conv2
		+ conv2Area * c
		+ skip2 * conv2Size2;

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
					int xx = x + i - padding;
					int yy = y + j - padding;
					if(xx>= 0 && xx < pool1Size && yy >= 0 && yy < pool1Size)
						val += pl1[xx * pool1Size + yy] * w[i * kernelSize + j];
				}
			}
			cv2[idx] = d_nonLinearity(val + b, NONLIN);
		}
	}
}


/*
* function: get convolution layer and pooling output
* blocks  : dim3(batch, kernelAmount2, Config::instance()->getChannels()),
* threads : dim3(threadidx)
*/

__global__ void g_cfm_conv_2(
	double* pool1,
	double** arrayW,
	double** arrayB,
	double* conv2,
	int pool1Size,
	int kernelSize,
	int padding,
	int conv2Size,
	int k1Amount,
	int k2Amount,
	int pool1Area,
	int conv2Area,
	int numOfCFM,
	int NONLIN)
{
	int sp = blockIdx.x;
	int k2 = blockIdx.y;
	int c  = blockIdx.z;

	double* w  = arrayW[k2] + kernelSize * kernelSize * c;
	double  b  = arrayB[k2][c];

	int pool1Size2 = pool1Size * pool1Size;
	int conv2Size2 = conv2Size * conv2Size;

	int skip2 = sp * k2Amount + k2;

	double* cv2 = conv2
		+ conv2Area * c
		+ skip2 * conv2Size2;

	for(int tidx = 0; tidx < conv2Size2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < conv2Size2)
		{
			int x = idx / conv2Size;
			int y = idx % conv2Size;
			double val = 0.0;
			for(int k1 = 0; k1 < numOfCFM; k1++)
			{
				int kk1 = (k1 + k2) % k1Amount;
				double* pl1 = pool1
					+ pool1Area * c
					+ (sp * k1Amount + kk1) * pool1Size2;

				for (int i = 0; i < kernelSize; i++) {
					for (int j = 0; j < kernelSize; j++) {
						int xx = x + i - padding;
						int yy = y + j - padding;
						if(xx>= 0 && xx < pool1Size && yy >= 0 && yy < pool1Size)
							val += pl1[xx * pool1Size + yy] * w[i * kernelSize + j];
					}
				}
			}
			cv2[idx] = d_nonLinearity(val + b, NONLIN);
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

void convAndPooling()
{
	if(Config::instance()->getCFM() == 0){
		for (int i = 0; i < convNCFM.size(); i++) {
			convNCFM[i]->feedforward();
			poolings[i]->feedforward();
		}
	}else {
		for (int i = 0; i < convCFM.size(); i++) {
			convCFM[i]->feedforward();
			poolings[i]->feedforward();
		}
	}
}
/*
* blocks  : cuFullConnectActi[hl]->rows;
* threads : dim3(min(512, len));
*/
__global__ void g_FullConnectLayerActi(double* acti, double* b, int NumofNeurons, int NONLIN)
{
	double* data  = acti + blockIdx.x * NumofNeurons;
	for(int id = 0; id < NumofNeurons; id += blockDim.x)
	{
		int idx = id + threadIdx.x;
		if(idx < NumofNeurons)
		{
			double val = data[idx];
			val = val + b[idx];
			data[idx] = d_nonLinearity(val, NONLIN);
		}
	}
}
__global__ void g_dropW(double * w, double * dropW, double* afterDropW, int len)
{
	for(int i = 0; i < len; i += blockDim.x * gridDim.x)
	{
		int id = i + blockIdx.x * blockDim.x + threadIdx.x;
		if(id < len)
		{
			afterDropW[id] = dropW[id] * w[id];
		}
	}
}
/*
* function: cuMatrix(batch, size, channel) to cuMatrix(batch, size * channel, 1)
* blocks  : dim3(batch)
* threads : dim3(min(512, cuPool[poolidx]->cols))
*/
__global__ void g_cuPoolToFlActi(double* cuPool, double*cuPoolToFlActi, int batch, int size, int channel)
{
	int b   = blockIdx.x;
	int len = size * channel;
	for(int i = 0; i < len; i+=blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			int s = id / channel;
			int c = id % channel;
			cuPoolToFlActi[b * size * channel + size * c + s] = cuPool[c * batch * size + b * size + s];
		}
	}
}

void getFullConnectLayerActi()
{
	for(int i = 0; i < fullConnect.size(); i++){
		fullConnect[i]->feedforward();
	}
}
/*
* blocks : cuSoftMaxP->rows
* threads: cuSoftMaxP->cols
* shared : sizeof(double) * cuSoftMaxP->cols * 2
*/
__global__ void g_getSoftMaxP(double* softMaxP, double* b, int cols)
{
	int bid = blockIdx.x;
	extern __shared__ double _share[];
	double * _max = _share;
	double * _sum = _share + blockDim.x;
	double* sp = softMaxP + bid * cols;
	_sum[threadIdx.x] = 0.0;
	_max[threadIdx.x] = -100000000.0;
	for(int tid = 0; tid < cols; tid += blockDim.x){
		int id = tid + threadIdx.x;
		if(id < cols){
			sp[id] += b[id];
			_max[threadIdx.x] = max(_max[threadIdx.x], sp[id]);
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
			if(_max[threadIdx.x] < _max[threadIdx.x + skip])
			{
				_max[threadIdx.x] = _max[threadIdx.x + skip];
			}
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();
	for(int tid = 0; tid < cols; tid += blockDim.x){
		int id = tid + threadIdx.x;
		if(id < cols){
			sp[id] -= _max[0];
			sp[id] = exp(sp[id]);
			_sum[threadIdx.x] += sp[id];
		}
	}
	__syncthreads();
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
	for(int tid = 0; tid < cols; tid += blockDim.x){
		int id = tid + threadIdx.x;
		if(id < cols){
			sp[id] /= _sum[0];
		}
	}
}
void getSoftMaxP(cuSMR& smr, cublasHandle_t handle)
{
	matrixMulTB(fullConnect[fullConnect.size() - 1]->getOutputs(),
		smr.Weight, cuSoftMaxP);
	
	int threads = min(512, cuSoftMaxP->cols);
	g_getSoftMaxP<<<cuSoftMaxP->rows, threads, sizeof(double) * threads * 2>>>(cuSoftMaxP->devData, smr.b->devData, cuSoftMaxP->cols);
	cudaDeviceSynchronize();
	getLastCudaError("g_getSoftMaxP");
}
/*
* function: getcost
*/
__global__ void g_getCost_1(double* softMaxP,
	double* groundTruth, double* cost, int*y, int rows, int cols, int batch)
{
	extern __shared__ double _sum[];
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


void getCost(
	int*y,
	cuSMR &smr,
	int batch)
{
	g_getCost_1<<<dim3(1), dim3(256), sizeof(double) * 256>>>(cuSoftMaxP->devData, cuGroundTruth->devData,
		smr.cost->devData, y, cuSoftMaxP->rows, cuSoftMaxP->cols, batch);
	cudaDeviceSynchronize();
	getLastCudaError("g_getCost_1");

	g_getCost_2<<<dim3(1), dim3(256), sizeof(double) * 256>>>(smr.cost->devData,  smr.Weight->devData, Config::instance()->getSoftMax()[0]->m_weightDecay,
		smr.Weight->getLen());
	cudaDeviceSynchronize();
	getLastCudaError("g_getCost_2");

	/*full connnect layers*/
	for(int h = 0; h < fullConnect.size(); h++){
		fullConnect[h]->getCost(smr.cost);
	}

	if(Config::instance()->getCFM() == 0){
		for(int cl = 0; cl < convNCFM.size(); cl++)
		{
			convNCFM[cl]->getCost(smr.cost);
		}
	}
	else{
		for(int cl = 0; cl < convCFM.size(); cl++){
			convCFM[cl]->getCost(smr.cost);
		}
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
*/
__global__ void g_getSmrWgrad(double* wgrad, double* weight, double lambda, int len, int batch)
{
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			wgrad[id] = lambda * weight[id] + wgrad[id] / batch;;
		}
	}
}



void getSoftMaxDelta(cuSMR &smr, double lambda, int batch, cublasHandle_t handle)
{
	g_getSoftMaxDelta<<<dim3(1), dim3(256)>>>(cuSoftMaxDelta->devData,
		cuSoftMaxP->devData,
		cuGroundTruth->devData, cuSoftMaxDelta->getLen());
	cudaDeviceSynchronize();
	matrixMulTA(cuSoftMaxDelta, fullConnect[fullConnect.size() - 1]->getOutputs(), smr.Wgrad);
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

__global__ void g_getFullConnectWgrad(double* wgrad, double* w, double* dropM, int len, double lambda, int batch)
{
	for(int i = 0; i < len; i += blockDim.x * gridDim.x)
	{
		int id = i + blockDim.x * blockIdx.x + threadIdx.x;
		if(id < len)
		{
			if(fabs(lambda) < 1e-10)
				wgrad[id] = wgrad[id] / batch * dropM[id];
			else
				wgrad[id] = (wgrad[id] / batch + lambda * w[id]) * dropM[id];
		}
	}
}
void getFullConnectDelta(
	cuSMR &smr,
	int batch,
	cublasHandle_t handle)
{
	matrixMul(cuSoftMaxDelta,
		smr.Weight, fullConnect[fullConnect.size() - 1]->getCurDelta());
	for(int hl = Config::instance()->getFC().size() - 1; hl >= 0; hl--)
	{
		if(hl == Config::instance()->getFC().size() - 1)
		{
			fullConnect[hl]->backpropagation();
		}
		else
		{
			fullConnect[hl]->backpropagation();
		}
	}
	for(int hl = Config::instance()->getFC().size() - 1; hl >= 0; hl--)
	{
		fullConnect[hl]->getGrad();
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
	int len = size * channels;
	for(int i = 0; i < len; i += blockDim.x)
	{
		int id = i + threadIdx.x;
		if(id < len)
		{
			int s = id / channels;
			int c = id % channels;
			cuPoolDelta[c * batch * size + b * size + s] = cuPoolFlDelta[b * size * channels + size * c + s];
		}
	}
}

void dConvAndUnpooling(double**x, 
	int batch, int ImgSize, int nclasses, cublasHandle_t handle)
{
	for(int cl = convNCFM.size() - 1; cl >= 0; cl--)
	{
		poolings[cl]->backpropagation();
		convNCFM[cl]->backpropagation();
		convNCFM[cl]->getGrad();
	}
}



/*
* function:get the first convolution lay bias grad
* blocks  :dim3(kernelAmount2, Config::instance()->getChannels())
* threads :dim3(256),
*/
__global__ void g_cfm_getCLayerBgrad_1(double* delta,
	double** bgrad,
	int deltaSize,
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
				+ deltaSize2 * (s * kernelAmount2 + k2)
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

void cfm_dConvAndUnpooling(double**x,
	int batch, int ImgSize, int nclasses, cublasHandle_t handle)
{
	for(int cl = convCFM.size() - 1; cl >= 0; cl--)
	{
		poolings[cl]->backpropagation();
		convCFM[cl]->backpropagation();
		convCFM[cl]->getGrad();
	}
}


void updataWB(
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
	
	for(int i = 0; i < fullConnect.size(); i++)
	{
		fullConnect[i]->updateWeight();
	}

	if(Config::instance()->getCFM() == 0){
		for(int cl = 0; cl < convNCFM.size(); cl++)
		{
			convNCFM[cl]->updateWeight();
		}
	}else {
		for(int cl = 0; cl < convCFM.size(); cl++)
		{
			convCFM[cl]->updateWeight();
		}
	}
	cudaDeviceSynchronize();
	getLastCudaError("updateWB");
}
void getNetworkCost(double** x, 
	int* y, 
	cuSMR &smr,
	int batch,
	int ImgSize, 
	int nclasses,
	cublasHandle_t handle)
{
	convAndPooling();
	getFullConnectLayerActi();
	getSoftMaxP(smr, handle);
	getCost(y,smr, batch);
	getSoftMaxDelta(smr,Config::instance()->getSoftMax()[0]->m_weightDecay, batch, handle);
	getFullConnectDelta(smr, batch, handle);
	if(Config::instance()->getCFM())
		cfm_dConvAndUnpooling(x, batch, ImgSize, nclasses, handle);
	else
		dConvAndUnpooling(x, batch, ImgSize, nclasses, handle);
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
	cuSMR &smr, int batch, int ImgSize, int nclasses, cublasHandle_t handle)
{
	convAndPooling();
	getFullConnectLayerActi();
	getSoftMaxP(smr, handle);
	g_getCorrect<<<dim3(1), batch>>>(
		cuSoftMaxP->devData, 
		cuSoftMaxP->cols,
		vote);
	cudaDeviceSynchronize();
}
void gradientChecking(cuSMR &smr, double**x, 
	int*y, int batch, int ImgSize, int nclasses, cublasHandle_t handle)
{
	/*for(int hl = 0; hl < hLayers.size(); hl++)
	{
		dropDelta(hLayers[hl].dropW, Config::instance()->getFC()[hl]->m_dropoutRate);
	}
	std::cout<<"test network !!!!"<<std::endl;
	double epsilon = 1e-4;
	for(int a = 0; a < convNCFM.size(); a++)
	{
		for(int b = 0; b < CLayers[a].layer.size(); b++)
		{
			printf("====%d %d\n",a, b);
			getNetworkCost(x,
				y,
				CLayers, hLayers,
				smr,
				batch, ImgSize, nclasses, handle);
			CLayers[a].layer[b].Wgrad->toCpu();
			cuMatrix<double>* grad = new cuMatrix<double>(CLayers[a].layer[b].Wgrad->hostData, CLayers[a].layer[b].Wgrad->rows,
				CLayers[a].layer[b].Wgrad->cols, CLayers[a].layer[b].Wgrad->channels);
			for(int c = 0; c < CLayers[a].layer[b].W->channels; c++){
				for(int i = 0; i < CLayers[a].layer[b].W->rows; i++){
					for(int j = 0; j < CLayers[a].layer[b].W->cols; j++){
						double memo = CLayers[a].layer[b].W->get(i, j, c);
						CLayers[a].layer[b].W->set(i, j, c, memo + epsilon);
						CLayers[a].layer[b].W->toGpu();
						getNetworkCost(x, y, CLayers, hLayers, smr, batch, ImgSize, nclasses, handle);
						smr.cost->toCpu();
						double value1 = smr.cost->get(0, 0 , 0);
						CLayers[a].layer[b].W->set(i, j, c, memo - epsilon);
						CLayers[a].layer[b].W->toGpu();
						getNetworkCost(x, y, CLayers, hLayers, smr, batch, ImgSize, nclasses, handle);
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
	}*/
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
	cuSMR &smr,
	cuMatrixVector<double>&testX,
	cuMatrix<int>* testY,
	int batch,
	int ImgSize,
	int nclasses,
	cublasHandle_t handle) {
		for(int hl = 0; hl < fullConnect.size(); hl++){
			fullConnect[hl]->drop(0.0);
		}

		cuVote->gpuClear();
		int cropr[] = { Config::instance()->getCrop() / 2, 0, 0,
			Config::instance()->getCrop(), Config::instance()->getCrop() };
		int cropc[] = { Config::instance()->getCrop() / 2, 0,
			Config::instance()->getCrop(), 0, Config::instance()->getCrop() };
		//	double scale[] = {0.0, -Config::instance()->getScale(), Config::instance()->getScale()};
		//	double rotation[] = {0.0, -Config::instance()->getRotation(), Config::instance()->getRotation()};
	
		cudaStream_t stream1;
		checkCudaErrors(cudaStreamCreate(&stream1));
		for (int h = 0; h < (Config::instance()->getHorizontal() == true ? 2 : 1); h++) {
				for (int c = 0; c < (Config::instance()->getCrop() == 0 ? 1 : 5); c++) {
					int batchImgId = 1;
					getBatchImageWithStreams(testX, batchImg[0], 0, stream1);
					for (int p = 0; p < testX.size() / batch; p++) {
						cudaStreamSynchronize(stream1);
						printf("test  %2d%%", 100 * p / ((testX.size() + batch - 1) / batch));
						int tstart = p * batch;
						if(tstart + batch <= testX.size() - batch)
							getBatchImageWithStreams(testX, batchImg[batchImgId], tstart + batch, stream1);
						
						batchImgId = 1 - batchImgId;
						cuApplyCrop(batchImg[batchImgId].m_devPoint,
							cu_distortion_vector->m_devPoint, batch, ImgSize,
							cropr[c], cropc[c]);
						if (h == 1)
							cuApplyHorizontal(cu_distortion_vector->m_devPoint,
							cu_distortion_vector->m_devPoint, batch, ImgSize, HORIZONTAL);
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
							cuVote->devData + tstart * nclasses,
							smr, batch, ImgSize, nclasses,
							handle);
						printf("\b\b\b\b\b\b\b\b\b");
					}
				}
		}
		checkCudaErrors(cudaStreamDestroy(stream1));
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
			
			cuSaveConvNet(smr);
		}
}


int voteTestDate(
	cuSMR &smr,
	cuMatrixVector<double>&testX,
	cuMatrix<int>* testY,
	cuMatrix<int>*& vote,
	int batch,
	int ImgSize,
	int nclasses,
 	cublasHandle_t handle) {
		for (int hl = 0; hl < Config::instance()->getFC().size(); hl++) {
			fullConnect[hl]->drop(0.0);
		}
		vote->gpuClear();
		int cropr[] = { Config::instance()->getCrop() / 2, 0, 0,
			Config::instance()->getCrop(), Config::instance()->getCrop() };
		int cropc[] = { Config::instance()->getCrop() / 2, 0,
			Config::instance()->getCrop(), 0, Config::instance()->getCrop() };
		for (int h = 0; h < (Config::instance()->getHorizontal() == true ? 2 : 1);
			h++) {
				for (int c = 0; c < (Config::instance()->getCrop() == 0 ? 1 : 5); c++) {
					for (int p = 0; p < testX.size() / batch; p++) {
						printf("test  %2d%%",
							100 * p / ((testX.size() + batch - 1) / batch));
						int tstart = p * batch;
						cuApplyCrop(testX.m_devPoint + tstart,
							cu_distortion_vector->m_devPoint, batch, ImgSize,
							cropr[c], cropc[c]);
						if (h == 1)
							cuApplyHorizontal(cu_distortion_vector->m_devPoint,
							cu_distortion_vector->m_devPoint, batch, ImgSize, HORIZONTAL);
						resultProdict(cu_distortion_vector->m_devPoint,
							testY->devData + tstart,
							vote->devData + tstart * nclasses,
							smr, batch, ImgSize, nclasses,
							handle);
						printf("\b\b\b\b\b\b\b\b\b");
					}
				}
		}
		cuCorrect->gpuClear();
		g_getVotingResult<<<dim3((testX.size() + batch - 1) / batch), dim3(batch)>>>(
			vote->devData,
			testY->devData,
			cuCorrect->devData,
			testX.size(),
			nclasses);
		cudaDeviceSynchronize();
		getLastCudaError("g_getVotingResult");
		cuCorrect->toCpu();
		return cuCorrect->get(0,0,0);
}


void getBatchImageWithStreams(cuMatrixVector<double>&x, cuMatrixVector<double>&batchImg, int start, cudaStream_t stream1){
	 for(int i = 0; i < batchImg.size(); i++){
		 memcpy(batchImg[i]->hostData, x[i + start]->hostData, sizeof(double) * batchImg[i]->getLen());
		 batchImg[i]->toGpu(stream1);
	 }
}


void cuTrainNetwork(cuMatrixVector<double>&x,
	cuMatrix<int>*y,
	cuSMR &smr,
	cuMatrixVector<double>&testX,
	cuMatrix<int>* testY,
	int batch,
	int ImgSize,
	int nclasses,
	std::vector<double>&nlrate,
	std::vector<double>&nMomentum,
	std::vector<int>&epoCount,
	cublasHandle_t handle)
{
	if(nlrate.size() != nMomentum.size() || nMomentum.size() != epoCount.size() || nlrate.size() != epoCount.size())
	{
		printf("nlrate, nMomentum, epoCount size not equal\n");
		exit(0);
	}

	if(Config::instance()->getIsGradientChecking())
		gradientChecking(smr, x.m_devPoint, y->devData, batch, ImgSize, nclasses, handle);


	predictTestDate(x, y, smr, testX, testY, batch, ImgSize, nclasses, handle);
	printf("correct is %d\n", cuCorrect->get(0,0,0));

	int epochs = 10000;

	double lrate = 0.05;
	double Momentum = 0.9;
	int id = 0;
	for (int epo = 0; epo < epochs; epo++) {
		if (id >= nlrate.size())
			break;
		lrate = nlrate[id];
		Momentum = nMomentum[id];
		Config::instance()->setLrate(lrate);
		Config::instance()->setMomentum(Momentum);

		double start, end;
		start = clock();
		cuApplyRandom(batch, clock(), ImgSize);
		for (int hl = 0; hl < fullConnect.size(); hl++) {
			fullConnect[hl]->drop();
		}

		x.shuffle(500, y);

		cudaStream_t stream1;
		checkCudaErrors(cudaStreamCreate(&stream1));

		getBatchImageWithStreams(x, batchImg[0], 0, stream1);
		int batchImgId = 1;
		for (int k = 0; k <= x.size() - batch; k += batch) {
			cudaStreamSynchronize(stream1);
			int start = k;
			printf("train %2d%%", 100 * k / ((x.size() + batch - 1)));
			if(start + batch <= x.size() - batch)
				getBatchImageWithStreams(x, batchImg[batchImgId], start + batch, stream1);
			batchImgId = 1 - batchImgId;
			//showImg(batchImg[batchImgId][batchImgId], 10);
			//cv::waitKey(0);
			
			cuApplyCropRandom(batchImg[batchImgId].m_devPoint,
				cu_distortion_vector->m_devPoint, batch, ImgSize);

			if(fabs(Config::instance()->getDistortion()) >= 0.1)
				cuApplyDistortion(cu_distortion_vector->m_devPoint, cu_distortion_vector->m_devPoint, batch, ImgSize);

			if (Config::instance()->getHorizontal()) {
				cuApplyHorizontal(cu_distortion_vector->m_devPoint,
					cu_distortion_vector->m_devPoint, batch, ImgSize, RANDOM_HORIZONTAL);
			}
			if (Config::instance()->getImageShow()) {
				for (int ff = batch - 1; ff >= 0; ff--) {
					showImg(batchImg[batchImgId][ff], 10);
					showImg(cu_distortion_vector->m_vec[ff], 10);
					cv::waitKey(0);
				}
			}

			getNetworkCost(cu_distortion_vector->m_devPoint, y->devData + start,
				smr, batch, ImgSize, nclasses,
				handle);
			updataWB(smr, lrate, Momentum, batch);
			printf("\b\b\b\b\b\b\b\b\b");
		}
		checkCudaErrors(cudaStreamDestroy(stream1));

		smr.cost->toCpu();
		char str[512];
		predictTestDate(x, y, smr, testX, testY,
			batch, ImgSize, nclasses, handle);
		if (epo && epo % epoCount[id] == 0) {
			cu_v_smr_w->gpuClear();
			cu_v_smr_b->gpuClear();

			for (int i = 0; i < fullConnect.size(); i++) {
				fullConnect[i]->clearMomentum();
			}
			if(Config::instance()->getCFM() == 0){
				for (int cl = 0; cl < convNCFM.size(); cl++) {
					convNCFM[cl]->clearMomentum();
				}
			}
			else 
			{
				for (int cl = 0; cl < convCFM.size(); cl++) {
					convCFM[cl]->clearMomentum();
				}
			}

			id++;
		}

		end = clock();
		sprintf(str, "e=%d t=%.03lfs cst=%lf crt=%d/%d mom=%.06lf r=%.08lf",
			epo, (double) (end - start) / CLOCKS_PER_SEC,
			smr.cost->get(0, 0, 0), cuCorrect->get(0, 0, 0), cuCurCorrect,
			Config::instance()->getMomentum(), Config::instance()->getLrate());
		printf("%s\n", str);
		LOG(str, "Result/log.txt");
	}
}


/*
*/
void __global__ g_getVoteAdd(int* voting, int* predict, int* y, int* correct, int len, int nclasses)
{
	for(int i = 0; i < len; i += blockDim.x * gridDim.x)
	{
		int idx = i + blockDim.x * blockIdx.x + threadIdx.x;
		if(idx < len)
		{
			int* pvoting = voting + idx * nclasses;
			int* ppredict= predict+ idx * nclasses;


			int _max = pvoting[0] + ppredict[0];
			int rid  = 0;
			for(int j = 0; j < nclasses; j++)
			{
				pvoting[j] += ppredict[j];
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

int cuVoteAdd(cuMatrix<int>*& voteSum, 
	cuMatrix<int>*& predict,
	cuMatrix<int>*& testY, 
	cuMatrix<int>*& correct,
	int nclasses)
{
	g_getVoteAdd<<<dim3((testY->getLen() + 256 - 1) / 256), dim3(256)>>>(
		voteSum->devData,
		predict->devData,
		testY->devData,
		correct->devData,
		testY->getLen(),
		nclasses);
	cudaDeviceSynchronize();
	getLastCudaError("g_getVoteAdd");
	correct->toCpu();
	return correct->get(0, 0, 0);
}
