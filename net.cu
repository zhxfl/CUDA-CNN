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
#include "layers/One.h"
#include "layers/BranchLayer.h"
#include "layers/CombineLayer.h"
#include "layers/DataLayer.h"
#include "layers/NIN.h"

#include <queue>
#include <set>

int cuCurCorrect;
cuMatrix<int>*cuCorrect = NULL;
cuMatrix<int>*cuVote = NULL;
std::vector<ConfigBase*>que;

void cuSaveConvNet()
{	
	FILE *pOut = fopen("Result/checkPoint.txt", "w");
	for(int i = 0; i < (int)que.size(); i++){
		LayerBase* layer = Layers::instance()->get(que[i]->m_name);
		layer->save(pOut);
	}
	fclose(pOut);
};

void cuFreeConvNet()
{
}

void cuReadConvNet(
	int imgDim,
    const char* path,
	int nclasses)
{	
	FILE *pIn = fopen(path, "r");

	for(int i = 0; i < (int)que.size(); i++){
		LayerBase* layer = Layers::instance()->get(que[i]->m_name);
		layer->initFromCheckpoint(pIn);
	}

	fclose(pIn);
};

void buildNetWork(int trainLen, int testLen)
{
	/*BFS*/
	std::queue<ConfigBase*>qqq;
	std::set<ConfigBase*> inque;
	for(int i = 0; i < (int)Config::instance()->getFirstLayers().size(); i++){
		qqq.push(Config::instance()->getFirstLayers()[i]);
		inque.insert(Config::instance()->getFirstLayers()[i]);
	}

	char logStr[1024];
	sprintf(logStr, "\n\n******************layer nexts start********************\n");
	LOG(logStr, "Result/log.txt");
	std::set<ConfigBase*>finish;
	while(!qqq.empty()){
		ConfigBase* top = qqq.front();
		qqq.pop();
		finish.insert(top);
		que.push_back(top);

		if(top->m_type == std::string("CONV")){
			ConfigConv * conv = (ConfigConv*) top;
			new ConvCFM(conv->m_name);
		}else if(top->m_type == std::string("LOCAL")){
			 new LocalConnect(top->m_name);
		}else if(top->m_type == std::string("BRANCHLAYER")){
			new BranchLayer(top->m_name);
		}else if(top->m_type == std::string("COMBINELAYER")){
			ConfigCombineLayer *bl = static_cast<ConfigCombineLayer*>(top);
			bool flag = true;
			for(int i = 0; i < (int)bl->m_inputs.size(); i++){
				ConfigBase* cb = Config::instance()->getLayerByName(bl->m_inputs[i]);
				if(finish.find(cb) == finish.end()){
					qqq.push(top);
					flag = false;
					finish.erase(top);
					break;
				}
			}
			if(flag == false) continue;
			else new CombineLayer(top->m_name);
		}else if(top->m_type == std::string("POOLING")){
			new Pooling(top->m_name);
		}else if(top->m_type == std::string("FC")){
			new FullConnect(top->m_name);
		}else if(top->m_type == std::string("SOFTMAX")){
			new SoftMax(top->m_name);
		}else if(top->m_type == std::string("ONE")){
			new One(top->m_name);
		}else if(std::string("LRN") == top->m_type){
			new LRN(top->m_name);
		}else if(std::string("DATA") == top->m_type){
			new DataLayer(top->m_name);
		}else if(std::string("NIN") == top->m_type){
			new NIN(top->m_name);
		}

		sprintf(logStr, "layer %15s:", top->m_name.c_str());
		LOG(logStr, "Result/log.txt");
		for(int n = 0; n < (int)top->m_next.size(); n++){
			if(inque.find(top->m_next[n]) == inque.end()){
				qqq.push(top->m_next[n]);
				inque.insert(top->m_next[n]);
			}
			sprintf(logStr, "%s ", top->m_next[n]->m_name.c_str());
			LOG(logStr, "Result/log.txt");
		}sprintf(logStr, "\n");
		LOG(logStr, "Result/log.txt");
	}

	sprintf(logStr, "\n\n******************layer nexts end********************\n");
	LOG(logStr, "Result/log.txt");
	/*correct and cuVote*/
	if(cuCorrect == NULL)
	{
		cuCorrect = new cuMatrix<int>(1,1,1);
		cuVote    = new cuMatrix<int>(testLen, Config::instance()->getClasses(), 1);
	}
}

void cuFreeCNNMemory(
	int batch,
	cuMatrixVector<float>&trainX, 
	cuMatrixVector<float>&testX)
{
}

void updataWB()
{
	/*updateWb*/
	for(int i = 0; i < (int)que.size(); i++){
		LayerBase* layer = Layers::instance()->get(que[i]->m_name);
		layer->updateWeight();
	}
	cudaDeviceSynchronize();
	getLastCudaError("updateWB");
}

void getNetworkCost(int* y)
{
	/*feedforward*/
	for(int i = 0; i < (int)que.size(); i++){
		if(que[i]->m_type == std::string("SOFTMAX")){
			SoftMax* sm = (SoftMax*)Layers::instance()->get(que[i]->m_name);
			sm->setPredict(y);
		}
	}
	

	for(int i = 0; i < (int)que.size(); i++){
		LayerBase* layer = Layers::instance()->get(que[i]->m_name);
		layer->feedforward();
	}

	/*backpropagation*/
	for(int i = (int)que.size() - 1; i >=0; i--){
		ConfigBase* top = que[i];
		LayerBase* layer = Layers::instance()->get(top->m_name);
		layer->backpropagation();
		layer->getGrad();
	}
}

/*
dim3(1),dim3(batch)
*/
__global__ void g_getCorrect(float* softMaxP, int cols,  int start, int* vote)
{
	int id = threadIdx.x;
	if(id < start)return;
	float* p = softMaxP + id * cols;
	int* votep= vote     + id * cols;

	int r = 0;
	float maxele = log(p[0]);
	for(int i = 1; i < cols; i++)
	{
		float val = log(p[i]);
		if(maxele < val)
		{
			maxele = val;
			r = i;
		}
	}
	votep[r]++;
}

void resultProdict(int* vote,int start)
{
	/*feedforward*/

	for(int i = 0; i < (int)que.size(); i++){
		LayerBase* layer = Layers::instance()->get(que[i]->m_name);
		layer->feedforward();
	}

	for(int i = 0; i < (int)que.size(); i++){
		if(que[i]->m_type == std::string("SOFTMAX")){
			g_getCorrect<<<dim3(1), Config::instance()->getBatchSize()>>>(
				Layers::instance()->get(que[i]->m_name)->getOutputs()->getDev(),
				Layers::instance()->get(que[i]->m_name)->getOutputs()->cols,
				start,
				vote);
			cudaDeviceSynchronize();
		}
	}
}

void gradientChecking(float**x, 
	int*y, int batch, int ImgSize, int nclasses, cublasHandle_t handle)
{
	/*for(int hl = 0; hl < hLayers.size(); hl++)
	{
		dropDelta(hLayers[hl].dropW, Config::instance()->getFC()[hl]->m_dropoutRate);
	}
	std::cout<<"test network !!!!"<<std::endl;
	float epsilon = 1e-4;
	for(int a = 0; a < convNCFM.size(); a++)
	{
		for(int b = 0; b < CLayers[a].layer.size(); b++)
		{
			sprintf(logStr, "====%d %d\n",a, b);
			getNetworkCost(x,
				y,
				CLayers, hLayers,
				smr,
				batch, ImgSize, nclasses, handle);
			CLayers[a].layer[b].Wgrad->toCpu();
			cuMatrix<float>* grad = new cuMatrix<float>(CLayers[a].layer[b].Wgrad->getHost(), CLayers[a].layer[b].Wgrad->rows,
				CLayers[a].layer[b].Wgrad->cols, CLayers[a].layer[b].Wgrad->channels);
			for(int c = 0; c < CLayers[a].layer[b].W->channels; c++){
				for(int i = 0; i < CLayers[a].layer[b].W->rows; i++){
					for(int j = 0; j < CLayers[a].layer[b].W->cols; j++){
						float memo = CLayers[a].layer[b].W->get(i, j, c);
						CLayers[a].layer[b].W->set(i, j, c, memo + epsilon);
						CLayers[a].layer[b].W->toGpu();
						getNetworkCost(x, y, CLayers, hLayers, smr, batch, ImgSize, nclasses, handle);
						smr.cost->toCpu();
						float value1 = smr.cost->get(0, 0 , 0);
						CLayers[a].layer[b].W->set(i, j, c, memo - epsilon);
						CLayers[a].layer[b].W->toGpu();
						getNetworkCost(x, y, CLayers, hLayers, smr, batch, ImgSize, nclasses, handle);
						smr.cost->toCpu();
						float value2 = smr.cost->get(0, 0, 0);
						float tp = (value1 - value2) / (2 * epsilon);
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


void predictTestDate(cuMatrixVector<float>&x,
	cuMatrix<int>*y ,
	cuMatrixVector<float>&testX,
	cuMatrix<int>* testY,
	int batch,
	int ImgSize,
	int nclasses,
	bool vote,
	cublasHandle_t handle) {
		Config::instance()->setTraining(false);
	
		int cropr[] = {Config::instance()->getCrop() / 2, 0, 0, Config::instance()->getCrop(), Config::instance()->getCrop()};
		int cropc[] = {Config::instance()->getCrop() / 2, 0, Config::instance()->getCrop(), 0, Config::instance()->getCrop()};

		float scalex[] = {0, -Config::instance()->getScale(), Config::instance()->getScale()};
		float scaley[] = {0, -Config::instance()->getScale(), Config::instance()->getScale()};
		float rotate[] = {0, -Config::instance()->getRotation(), Config::instance()->getRotation()};

		int hlen = Config::instance()->getHorizontal() == 1 ? 2 : 1;
		int clen = Config::instance()->getCrop() == 0 ? 1 : sizeof(cropc) / sizeof(int);
		int scaleLen = Config::instance()->getScale() == 0 ? 1 : sizeof(scalex) / sizeof(float);
		int rotateLen = Config::instance()->getRotation() == 0 ? 1 : sizeof(rotate) / sizeof(float);
		if(!vote) hlen = clen = scaleLen = rotateLen = 1;
		
		DataLayer *dl = static_cast<DataLayer*>(Layers::instance()->get("data"));
		dl->getBatchImageWithStreams(x, 0);

		cuVote->gpuClear();
		for(int sidx = 0; sidx < scaleLen; sidx++){
			for(int sidy = 0; sidy < scaleLen; sidy++){
				for(int rid = 0; rid < rotateLen; rid++){
					for (int h = 0; h < hlen; h++) {
						for (int c = 0; c < clen; c++) {
							dl->getBatchImageWithStreams(testX, 0);
							for (int p = 0; p < ((int)testX.size() + batch - 1) / batch; p++) {
								dl->synchronize();
								printf("test  %2d%%", 100 * p / ((testX.size() + batch - 1) / batch));
								int tstart = p * batch;
								if(tstart + batch <= (int)testX.size() - batch)
									dl->getBatchImageWithStreams(testX, tstart + batch);
								else {
									int start = testX.size() - batch;
									dl->getBatchImageWithStreams(testX, start);
								}

								if(tstart + batch > (int)testX.size()){
									tstart = (int)testX.size() - batch;
								}

								dl->testData(cropr[c], cropc[c], rotate[rid], scalex[sidx], scaley[sidy], h);

								resultProdict(cuVote->getDev() + tstart * nclasses,
									p * batch - tstart);
								
								printf("\b\b\b\b\b\b\b\b\b");
							}
						}
					}
				}
			}
		}
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



void getBatchImageWithStreams(cuMatrixVector<float>&x,
	cuMatrixVector<float>&batchImg, 
	int start, 
	cudaStream_t stream1){
	 for(int i = 0; i < (int)batchImg.size(); i++){
		 memcpy(batchImg[i]->getHost(), x[i + start]->getHost(), sizeof(float) * batchImg[i]->getLen());
		 batchImg[i]->toGpu(stream1);
	 }
}

float getCost(){
	float cost = 0.0;
	for(int i = 0; i < (int)que.size(); i++){
		LayerBase* layer = (LayerBase*)Layers::instance()->get(que[i]->m_name);
		layer->calCost();
		layer->printCost();
		cost += layer->getCost();
	}
	return cost;
}

void cuTrainNetwork(cuMatrixVector<float>&x,
	cuMatrix<int>*y,
	cuMatrixVector<float>&testX,
	cuMatrix<int>* testY,
	int batch,
	int ImgSize,
	int nclasses,
	std::vector<float>&nlrate,
	std::vector<float>&nMomentum,
	std::vector<int>&epoCount,
	cublasHandle_t handle)
{
	char logStr[1024];
	if(nlrate.size() != nMomentum.size() || nMomentum.size() != epoCount.size() || nlrate.size() != epoCount.size())
	{
		printf("nlrate, nMomentum, epoCount size not equal\n");
		exit(0);
	}

	if(Config::instance()->getIsGradientChecking())
		gradientChecking(x.m_devPoint, y->getDev(), batch, ImgSize, nclasses, handle);


	predictTestDate(x, y, testX, testY, batch, ImgSize, nclasses, 0, handle);
	sprintf(logStr, "correct is %d\n", cuCorrect->get(0,0,0));
	LOG(logStr, "Result/log.txt");

	int epochs = 10000;

	float lrate = 0.05f;
	float Momentum = 0.9f;
	int id = 0;
	for (int epo = 0; epo < epochs; epo++) {
		if (id >= (int)nlrate.size())
			break;
		lrate = nlrate[id];
		Momentum = nMomentum[id];
		Config::instance()->setLrate(lrate);
		Config::instance()->setMomentum(Momentum);

		float start, end;
		start = (float)clock();
		cuApplyRandom(batch, clock() + epo, ImgSize);

		Config::instance()->setTraining(true);

		x.shuffle(5000, y);


		DataLayer *dl = static_cast<DataLayer*>(Layers::instance()->get("data"));
		dl->getBatchImageWithStreams(x, 0);

		for (int k = 0; k < ((int)x.size() + batch - 1) / batch; k ++) {
			dl->synchronize();
			int start = k * batch;
			printf("train %2d%%", 100 * start / ((x.size() + batch - 1)));
			
			if(start + batch <= (int)x.size() - batch)
				dl->getBatchImageWithStreams(x, start + batch);
			else{
				int tstart = x.size() - batch;
				dl->getBatchImageWithStreams(x, tstart);
			}
			if(start + batch > (int)x.size()){
				start = (int)x.size() - batch;
			}

			dl->trainData();
			getNetworkCost(y->getDev() + start);
			updataWB();
			printf("\b\b\b\b\b\b\b\b\b");
		}

		float cost = getCost();

		end = (float)clock();
		sprintf(logStr, "epoch=%d time=%.03lfs cost=%f Momentum=%.06lf lrate=%.08lf\n",
			epo, (float) (end - start) / CLOCKS_PER_SEC,
			cost,
			Config::instance()->getMomentum(), Config::instance()->getLrate());
		LOG(logStr, "Result/log.txt");

		if (epo && epo % epoCount[id] == 0) {
// 			for(int i = 0; i < que.size(); i++){
// 				LayerBase* layer = (LayerBase*)Layers::instance()->get(que[i]->m_name);
// 				layer->clearMomentum();
// 			}
			id++;
		}
		

		sprintf(logStr, "===================weight value================\n");
		LOG(logStr, "Result/log.txt");
		for(int i = 0; i < (int)que.size(); i++){
			LayerBase* layer = Layers::instance()->get(que[i]->m_name);
			layer->printParameter();
		}

		
		sprintf(logStr, "===================test Result================\n");
		LOG(logStr, "Result/log.txt");
		predictTestDate(x, y, testX, testY,
			batch, ImgSize, nclasses, false, handle);
		sprintf(logStr, "test %.2lf%%/%.2lf%%\n", 100.0 * cuCorrect->get(0, 0, 0) / testX.size(),
			100.0 * cuCurCorrect / testX.size());
		LOG(logStr, "Result/log.txt");

		if(epo && epo % Config::instance()->getTestEpoch() == 0){
			predictTestDate(x, y, testX, testY,
				batch, ImgSize, nclasses, true, handle);
			sprintf(logStr, "test voting correct %.2lf%%/%.2lf%%\n", 100.0 * cuCorrect->get(0, 0, 0) / testX.size(),
				100.0 * cuCurCorrect / testX.size());
			LOG(logStr, "Result/log.txt");
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


