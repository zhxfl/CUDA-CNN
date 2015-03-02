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
#include "layers/SoftMax.h"


cuMatrixVector<double>* cu_distortion_vector;

int cuCurCorrect;
cuMatrix<int>*cuCorrect = NULL;
cuMatrix<int>*cuVote = NULL;
cuMatrix<double>*cost = NULL;

std::vector<Pooling*>poolings;
std::vector<ConvCFM*>convCFM;
std::vector<ConvNCFM*>convNCFM;
std::vector<FullConnect*>fullConnect;
std::vector<SoftMax*>softMax;


/*batch size images*/
cuMatrixVector<double>batchImg[2];

void getBatchImageWithStreams(cuMatrixVector<double>&x, cuMatrixVector<double>&batchImg, int start, cudaStream_t stream1);
void outputMatrix(cuMatrix<double>* m);


void cuSaveConvNet()
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

	for(int s = 0; s < softMax.size(); s++){
		softMax[s]->save(pOut);
	}
	fclose(pOut);
};

void cuFreeConvNet()
{
}

void cuReadConvNet(
	int imgDim, char* path,
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

	for(int s = 0; s < softMax.size(); s++){
		softMax[s]->initFromCheckpoint(pIn);
	}

	fclose(pIn);
};

void cuInitCNNMemory(
	int batch,
	cuMatrixVector<double>& trainX, 
	cuMatrixVector<double>& testX,
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

	for(int i = 0; i < Config::instance()->getSoftMax().size(); i++){
		softMax.push_back(new SoftMax(fullConnect[fullConnect.size() - 1]->getOutputs(), 
			batch, 
			Config::instance()->getSoftMax()[i]->m_weightDecay,
			Config::instance()->getSoftMax()[i]->m_numClasses,
			Config::instance()->getNonLinearity()));

		softMax[i]->initRandom();
		softMax[i]->setPreDelta(fullConnect[fullConnect.size() - 1]->getCurDelta());

	}

	/*correct and cuVote*/
	if(cuCorrect == NULL)
	{
		cuCorrect = new cuMatrix<int>(1,1,1);
		cuVote    = new cuMatrix<int>(testX.size(), Config::instance()->getSoftMax()[0]->m_numClasses, 1);
		cost      = new cuMatrix<double>(1, 1, 1);
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
	cuMatrixVector<double>&testX)
{
	delete cu_distortion_vector;
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

void getFullConnectLayerActi()
{
	for(int i = 0; i < fullConnect.size(); i++){
		fullConnect[i]->feedforward();
	}
}

void getSoftMaxP(cublasHandle_t handle)
{
	for(int s = 0; s < softMax.size(); s++){
		softMax[s]->feedforward();
	}
}

void getCost(
	int*y,
	int batch)
{
	for(int s = 0; s < softMax.size(); s++){
		softMax[s]->getCost(cost, y);
	}

	/*full connnect layers*/
	for(int h = 0; h < fullConnect.size(); h++){
		fullConnect[h]->getCost(cost);
	}

	if(Config::instance()->getCFM() == 0){
		for(int cl = 0; cl < convNCFM.size(); cl++)
		{
			convNCFM[cl]->getCost(cost);
		}
	}
	else{
		for(int cl = 0; cl < convCFM.size(); cl++){
			convCFM[cl]->getCost(cost);
		}
	}
}

void getSoftMaxDelta(double lambda, int batch, cublasHandle_t handle)
{
	for(int s = 0; s < softMax.size(); s++){
		softMax[s]->backpropagation();
	}
	for(int s = 0; s < softMax.size(); s++){
		softMax[s]->getGrad();
	}
}

void getFullConnectDelta(
	int batch,
	cublasHandle_t handle)
{
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
	double lrate,
	double momentum,
	int batch)
{
	for(int s = 0; s < softMax.size(); s++){
		softMax[s]->updateWeight();
	}
	
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
	int batch,
	int ImgSize, 
	int nclasses,
	cublasHandle_t handle)
{
	convAndPooling();
	getFullConnectLayerActi();
	getSoftMaxP(handle);
	getCost(y,batch);
	getSoftMaxDelta(Config::instance()->getSoftMax()[0]->m_weightDecay, batch, handle);
	getFullConnectDelta(batch, handle);
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
	 int batch, int ImgSize, int nclasses, cublasHandle_t handle)
{
	convAndPooling();
	getFullConnectLayerActi();
	getSoftMaxP(handle);
	g_getCorrect<<<dim3(1), batch>>>(
		softMax[0]->getOutputs()->devData,
		softMax[0]->getOutputs()->cols,
		vote);
	cudaDeviceSynchronize();
}

void gradientChecking(double**x, 
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
							batch, ImgSize, nclasses,
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
			
			cuSaveConvNet();
		}
}


int voteTestDate(
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
							batch, ImgSize, nclasses,
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
		gradientChecking(x.m_devPoint, y->devData, batch, ImgSize, nclasses, handle);


	predictTestDate(x, y, testX, testY, batch, ImgSize, nclasses, handle);
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
				batch, ImgSize, nclasses,
				handle);
			updataWB(lrate, Momentum, batch);
			printf("\b\b\b\b\b\b\b\b\b");
		}
		checkCudaErrors(cudaStreamDestroy(stream1));

		cost->toCpu();
		char str[512];
		predictTestDate(x, y, testX, testY,
			batch, ImgSize, nclasses, handle);
		if (epo && epo % epoCount[id] == 0) {

			for(int i = 0; i < softMax.size(); i++){
				softMax[i]->clearMomentum();
			}

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
			cost->get(0, 0, 0), cuCorrect->get(0, 0, 0), cuCurCorrect,
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
