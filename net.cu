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
#include "layers/FullConnect.h"
#include "layers/SoftMax.h"
#include "layers/LayerBase.h"
#include "layers/LocalConnect.h"
#include "layers/LRN.h"
#include "layers/NIN.h"

#include <queue>

cuMatrixVector<double>* cu_distortion_vector;

int cuCurCorrect;
cuMatrix<int>*cuCorrect = NULL;
cuMatrix<int>*cuVote = NULL;
std::vector<ConfigBase*>que;

/*batch size images*/
cuMatrixVector<double>batchImg[2];

void getBatchImageWithStreams(cuMatrixVector<double>&x, cuMatrixVector<double>&batchImg, int start, cudaStream_t stream1);
void outputMatrix(cuMatrix<double>* m);


void cuSaveConvNet()
{	
	FILE *pOut = fopen("Result/checkPoint.txt", "w");
	for(int i = 0; i < que.size(); i++){
		LayerBase* layer = Layers::instance()->get(que[i]->m_name);
		layer->save(pOut);
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

	for(int i = 0; i < que.size(); i++){
		LayerBase* layer = Layers::instance()->get(que[i]->m_name);
		layer->initFromCheckpoint(pIn);
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
	Layers::instance()->setInputs(cu_distortion_vector);


	/*BFS*/
	std::queue<ConfigBase*>qqq;
	for(int i = 0; i < Config::instance()->getFirstLayers().size(); i++){
		qqq.push(Config::instance()->getFirstLayers()[i]);
	}

	while(!qqq.empty()){
		ConfigBase* top = qqq.front();
		qqq.pop();
		que.push_back(top);

		if(top->m_type == std::string("CONV")){
			ConfigConv * conv = (ConfigConv*) top;
			new ConvCFM(conv->m_name);
		}else if(top->m_type == std::string("LOCAL")){
			 new LocalConnect(top->m_name);
		}
		else if(top->m_type == std::string("POOLING")){
			new Pooling(top->m_name);
		}else if(top->m_type == std::string("FC")){
			new FullConnect(top->m_name);
		}else if(top->m_type == std::string("SOFTMAX")){
			new SoftMax(top->m_name);
		}else if(top->m_type == std::string("NIN")){
			new NIN(top->m_name);
		}
		else if(std::string("LRN") == top->m_type){
			new LRN(top->m_name);
		}
		for(int n = 0; n < top->m_next.size(); n++){
			qqq.push(top->m_next[n]);
		}
	}

	/*correct and cuVote*/
	if(cuCorrect == NULL)
	{
		cuCorrect = new cuMatrix<int>(1,1,1);
		cuVote    = new cuMatrix<int>(testX.size(), Config::instance()->getClasses(), 1);
	}

	/*double buffer for batch images*/
	int crop = Config::instance()->getCrop();
	for(int i = 0; i < 2; i ++){
		for(int j = 0; j < batch; j++){
			batchImg[i].push_back(new cuMatrix<double>(ImgSize + crop, ImgSize + crop, Config::instance()->getChannels()));
		}
		batchImg[i].toGpu();
	}
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
		printf("\n");
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

void updataWB(
	double lrate,
	double momentum,
	int batch)
{
	/*updateWb*/
	for(int i = 0; i < que.size(); i++){
		LayerBase* layer = Layers::instance()->get(que[i]->m_name);
		layer->updateWeight();
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
	/*feedforward*/
	SoftMax* sm = (SoftMax*)Layers::instance()->get("softmax1");
	sm->setPredict(y);

	for(int i = 0; i < que.size(); i++){
		LayerBase* layer = Layers::instance()->get(que[i]->m_name);
		layer->feedforward();
	}

	/*Cost*/
// 	for(int i = que.size() - 1; i >= 0; i--){
// 		LayerBase* layer = Layers::instance()->get(que[i]->m_name);
// 		layer->getCost(cost, y);
// 	}


	/*backpropagation*/
	for(int i = que.size() - 1; i >=0; i--){
		ConfigBase* top = que[i];
		LayerBase* layer = Layers::instance()->get(top->m_name);
		layer->backpropagation();
		layer->getGrad();
	}
}

/*
dim3(1),dim3(batch)
*/
__global__ void g_getCorrect(double* softMaxP, int cols,  int start, int* vote)
{
	int id = threadIdx.x;
	if(id < start)return;
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
	 int batch, int ImgSize, int nclasses, int start, cublasHandle_t handle)
{
	/*feedforward*/

	for(int i = 0; i < que.size(); i++){
		LayerBase* layer = Layers::instance()->get(que[i]->m_name);
		layer->feedforward();
	}

	g_getCorrect<<<dim3(1), batch>>>(
		Layers::instance()->get("softmax1")->getOutputs()->getDev(),
		Layers::instance()->get("softmax1")->getOutputs()->cols,
		start,
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
			cuMatrix<double>* grad = new cuMatrix<double>(CLayers[a].layer[b].Wgrad->getHost(), CLayers[a].layer[b].Wgrad->rows,
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
	bool vote,
	cublasHandle_t handle) {
		for(int i = 0; i < que.size(); i++){
			if(que[i]->m_type == std::string("FC")){
				FullConnect* layer = (FullConnect*)Layers::instance()->get(que[i]->m_name);
				layer->drop(0.0);
			}
		}

		cuVote->gpuClear();
		int cropr[] = {Config::instance()->getCrop() / 2, 0, 0, Config::instance()->getCrop(), Config::instance()->getCrop()};
		int cropc[] = {Config::instance()->getCrop() / 2, 0, Config::instance()->getCrop(), 0, Config::instance()->getCrop()};

		double scalex[] = {0, -Config::instance()->getScale(), Config::instance()->getScale()};
		double scaley[] = {0, -Config::instance()->getScale(), Config::instance()->getScale()};

		double rotate[] = {0, -Config::instance()->getRotation(), Config::instance()->getRotation()};
// 		if(fabs(Config::instance()->getDistortion()) >= 0.1 || Config::instance()->getScale() >= 1 || Config::instance()->getRotation() >= 1)
// 			cuApplyDistortion(cu_distortion_vector->m_devPoint, cu_distortion_vector->m_devPoint, batch, ImgSize);

		cudaStream_t stream1;
		checkCudaErrors(cudaStreamCreate(&stream1));

		int hlen = Config::instance()->getHorizontal() == 1 ? 2 : 1;
		int clen = Config::instance()->getCrop() == 0 ? 1 : sizeof(cropc) / sizeof(int);
		int scaleLen = Config::instance()->getScale() == 0 ? 1 : sizeof(scalex) / sizeof(double);
		int rotateLen = Config::instance()->getRotation() == 0 ? 1 : sizeof(rotate) / sizeof(double);
		if(!vote) hlen = clen = scaleLen = rotateLen = 1;
		
		for(int sidx = 0; sidx < scaleLen; sidx++){
			for(int sidy = 0; sidy < scaleLen; sidy++){
				for(int rid = 0; rid < rotateLen; rid++){
					cuApplyScaleAndRotate(batch, ImgSize, scalex[sidx], scaley[sidy], rotate[rid]);
					for (int h = 0; h < hlen; h++) {
						for (int c = 0; c < clen; c++) {
							int batchImgId = 1;
							getBatchImageWithStreams(testX, batchImg[0], 0, stream1);
							for (int p = 0; p < (testX.size() + batch - 1) / batch; p++) {
								cudaStreamSynchronize(stream1);
								printf("test  %2d%%", 100 * p / ((testX.size() + batch - 1) / batch));
								int tstart = p * batch;
								if(tstart + batch <= testX.size() - batch)
									getBatchImageWithStreams(testX, batchImg[batchImgId], tstart + batch, stream1);
								else {
									int start = testX.size() - batch;
									getBatchImageWithStreams(testX, batchImg[batchImgId], start, stream1);
								}

								if(tstart + batch > testX.size()){
									tstart = testX.size() - batch;
								}

								//printf("start = %d\n", tstart);
								batchImgId = 1 - batchImgId;
								cuApplyCrop(batchImg[batchImgId].m_devPoint,
									cu_distortion_vector->m_devPoint, batch, ImgSize,
									cropr[c], cropc[c]);

								cuApplyDistortion(cu_distortion_vector->m_devPoint, cu_distortion_vector->m_devPoint, batch, ImgSize);

								if (h == 1)
									cuApplyHorizontal(cu_distortion_vector->m_devPoint,
									cu_distortion_vector->m_devPoint, batch, ImgSize, HORIZONTAL);
								resultProdict(cu_distortion_vector->m_devPoint,
									testY->getDev() + tstart,
									cuVote->getDev() + tstart * nclasses,
									batch, ImgSize, nclasses, p * batch - tstart,
									handle);
								printf("\b\b\b\b\b\b\b\b\b");
							}
						}
					}
				}
			}
		}
		checkCudaErrors(cudaStreamDestroy(stream1));
		cuCorrect->gpuClear();
		g_getVotingResult<<<dim3((testX.size() + batch - 1) / batch), dim3(batch)>>>(
			cuVote->getDev(),
			testY->getDev(),
			cuCorrect->getDev(),
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

		for(int i = 0; i < que.size(); i++){
			if(que[i]->m_type == std::string("FC")){
				FullConnect* layer = (FullConnect*)Layers::instance()->get(que[i]->m_name);
				layer->drop(0.0);
			}
		}

		cuVote->gpuClear();
		int cropr[] = {Config::instance()->getCrop() / 2, 0, 0, Config::instance()->getCrop(), Config::instance()->getCrop()};
		int cropc[] = {Config::instance()->getCrop() / 2, 0, Config::instance()->getCrop(), 0, Config::instance()->getCrop()};

		cudaStream_t stream1;
		checkCudaErrors(cudaStreamCreate(&stream1));
		for (int h = 0; h < (Config::instance()->getHorizontal() == 1 ? 2 : 1); h++) {
			for (int c = 0; c < (Config::instance()->getCrop() == 0 ? 1 : sizeof(cropc) / sizeof(int)); c++) {
				int batchImgId = 1;
				getBatchImageWithStreams(testX, batchImg[0], 0, stream1);
				for (int p = 0; p < (testX.size() + batch - 1) / batch; p++) {
					cudaStreamSynchronize(stream1);
					printf("test  %2d%%", 100 * p / ((testX.size() + batch - 1) / batch));
					int tstart = p * batch;
					if(tstart + batch <= testX.size() - batch)
						getBatchImageWithStreams(testX, batchImg[batchImgId], tstart + batch, stream1);
					else {
						int start = testX.size() - batch;
						getBatchImageWithStreams(testX, batchImg[batchImgId], start, stream1);
					}

					if(tstart + batch > testX.size()){
						tstart = testX.size() - batch;
					}
					batchImgId = 1 - batchImgId;
					cuApplyCrop(batchImg[batchImgId].m_devPoint,
						cu_distortion_vector->m_devPoint, batch, ImgSize,
						cropr[c], cropc[c]);
					if (h == 1)
						cuApplyHorizontal(cu_distortion_vector->m_devPoint,
						cu_distortion_vector->m_devPoint, batch, ImgSize, HORIZONTAL);
					resultProdict(cu_distortion_vector->m_devPoint,
						testY->getDev() + tstart,
						cuVote->getDev() + tstart * nclasses,
						batch, ImgSize, nclasses, p * batch - tstart,
						handle);
					printf("\b\b\b\b\b\b\b\b\b");
				}
			}
		}
		cuCorrect->gpuClear();
		g_getVotingResult<<<dim3((testX.size() + batch - 1) / batch), dim3(batch)>>>(
			vote->getDev(),
			testY->getDev(),
			cuCorrect->getDev(),
			testX.size(),
			nclasses);
		cudaDeviceSynchronize();
		getLastCudaError("g_getVotingResult");
		cuCorrect->toCpu();
		return cuCorrect->get(0,0,0);
}


void getBatchImageWithStreams(cuMatrixVector<double>&x, cuMatrixVector<double>&batchImg, int start, cudaStream_t stream1){
	 for(int i = 0; i < batchImg.size(); i++){
		 memcpy(batchImg[i]->getHost(), x[i + start]->getHost(), sizeof(double) * batchImg[i]->getLen());
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
		gradientChecking(x.m_devPoint, y->getDev(), batch, ImgSize, nclasses, handle);


	predictTestDate(x, y, testX, testY, batch, ImgSize, nclasses, 0, handle);
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
		
		for(int i = 0; i < que.size(); i++){
			if(que[i]->m_type == std::string("FC")){
				FullConnect* layer = (FullConnect*)Layers::instance()->get(que[i]->m_name);
				layer->drop();
			}
		}

		x.shuffle(5000, y);

		cudaStream_t stream1;
		checkCudaErrors(cudaStreamCreate(&stream1));

		getBatchImageWithStreams(x, batchImg[0], 0, stream1);
		int batchImgId = 1;
		for (int k = 0; k < (x.size() + batch - 1) / batch; k ++) {
			cudaStreamSynchronize(stream1);
			int start = k * batch;
			printf("train %2d%%", 100 * start / ((x.size() + batch - 1)));
			
			if(start + batch <= x.size() - batch)
				getBatchImageWithStreams(x, batchImg[batchImgId], start + batch, stream1);
			else{
				int tstart = x.size() - batch;
				getBatchImageWithStreams(x, batchImg[batchImgId], tstart, stream1);
			}
			if(start + batch > x.size()){
				start = x.size() - batch;
			}

			batchImgId = 1 - batchImgId;
			
			cuApplyCropRandom(batchImg[batchImgId].m_devPoint,
				cu_distortion_vector->m_devPoint, batch, ImgSize);

			if(fabs(Config::instance()->getDistortion()) >= 0.1 || Config::instance()->getScale() >= 1 || Config::instance()->getRotation() >= 1)
				cuApplyDistortion(cu_distortion_vector->m_devPoint, cu_distortion_vector->m_devPoint, batch, ImgSize);

			if (Config::instance()->getHorizontal()) {
				cuApplyHorizontal(cu_distortion_vector->m_devPoint,
					cu_distortion_vector->m_devPoint, batch, ImgSize, RANDOM_HORIZONTAL);
			}
			
			cuApplyWhiteNoise(cu_distortion_vector->m_devPoint,
				cu_distortion_vector->m_devPoint, batch, ImgSize, Config::instance()->getWhiteNoise());

			if (Config::instance()->getImageShow()) {
				for (int ff = batch - 1; ff >= 0; ff--) {
					showImg(batchImg[batchImgId][ff], 5);
					showImg(cu_distortion_vector->m_vec[ff], 5);
					cv::waitKey(0);
				}
			}

			getNetworkCost(cu_distortion_vector->m_devPoint, y->getDev() + start,
				batch, ImgSize, nclasses,
				handle);
			updataWB(lrate, Momentum, batch);
			printf("\b\b\b\b\b\b\b\b\b");
		}
		checkCudaErrors(cudaStreamDestroy(stream1));

		double cost = 0.0;
		for(int i = 0; i < que.size(); i++){
			LayerBase* layer = (LayerBase*)Layers::instance()->get(que[i]->m_name);
			layer->calCost();
			layer->printCost();
			cost += layer->getCost();
		}

		char str[512];

		end = clock();
		sprintf(str, "epoch=%d time=%.03lfs cost=%lf Momentum=%.06lf lrate=%.08lf",
			epo, (double) (end - start) / CLOCKS_PER_SEC,
			cost,
			Config::instance()->getMomentum(), Config::instance()->getLrate());
		printf("%s\n", str);
		LOG(str, "Result/log.txt");

		if (epo && epo % epoCount[id] == 0) {
			for(int i = 0; i < que.size(); i++){
				LayerBase* layer = (LayerBase*)Layers::instance()->get(que[i]->m_name);
				layer->clearMomentum();
			}
			id++;
		}
		

		printf("===================weight value================\n");
		for(int i = 0; i < que.size(); i++){
			LayerBase* layer = Layers::instance()->get(que[i]->m_name);
			layer->printParameter();
		}

		
		printf("===================test Result================\n");
		predictTestDate(x, y, testX, testY,
			batch, ImgSize, nclasses, false, handle);
		sprintf(str, "test %.2lf%%/%.2lf%%", 100.0 * cuCorrect->get(0, 0, 0) / testX.size(),
			100.0 * cuCurCorrect / testX.size());
		printf("%s\n",str);
		LOG(str, "Result/log.txt");

		if(epo && epo % Config::instance()->getTestEpoch() == 0){
			predictTestDate(x, y, testX, testY,
				batch, ImgSize, nclasses, true, handle);
			sprintf(str, "test voting correct %.2lf%%/%.2lf%%", 100.0 * cuCorrect->get(0, 0, 0) / testX.size(),
				100.0 * cuCurCorrect / testX.size());
			printf("%s\n",str);
			LOG(str, "Result/log.txt");
		}

		
		if(epo == 0){
			MemoryMonitor::instance()->printCpuMemory();
			MemoryMonitor::instance()->printGpuMemory();
		}
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
		voteSum->getDev(),
		predict->getDev(),
		testY->getDev(),
		correct->getDev(),
		testY->getLen(),
		nclasses);
	cudaDeviceSynchronize();
	getLastCudaError("g_getVoteAdd");
	correct->toCpu();
	return correct->get(0, 0, 0);
}


