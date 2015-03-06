#include "SoftMax.h"
#include "../common/cuBase.h"
#include "../common/cuMatrix.h"
#include "../common/Config.h"
#include <math.h>

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

void SoftMax::feedforward()
{
	matrixMulTB(inputs,
		w, outputs);

	int threads = std::min(512, outputs->cols);
	g_getSoftMaxP<<<outputs->rows, threads, sizeof(double) * threads * 2>>>(
		outputs->devData,
		b->devData, 
		outputs->cols);
	cudaDeviceSynchronize();
	getLastCudaError("g_getSoftMaxP");
}

void SoftMax::backpropagation()
{
	g_getSoftMaxDelta<<<dim3(1), dim3(256)>>>(curDelta->devData,
		outputs->devData,
		groudTruth->devData, curDelta->getLen());
	cudaDeviceSynchronize();

	matrixMul(curDelta,
		w, preDelta);
}

void SoftMax::getGrad()
{
	matrixMulTA(curDelta, inputs, wgrad);

	g_getSmrWgrad<<<dim3(1), dim3(256)>>>(wgrad->devData,
		w->devData, lambda, wgrad->getLen(), batch);
	cudaDeviceSynchronize();

	if(curDelta->rows > MAX_THREADS)
	{
		printf("getSoftMaxDelta g_getBgrad > MAX_THREADS\n");
		exit(0);
	}
	g_getBgrad<<<dim3(curDelta->cols), dim3(curDelta->rows), 
		sizeof(double) * curDelta->rows>>>(
		curDelta->devData, 
		bgrad->devData,
		batch);
	cudaDeviceSynchronize();
	getLastCudaError("g_getBgrad");
}

void SoftMax::updateWeight()
{
	g_vecAdd<<<dim3(min((momentum_w->getLen() + 255) / 256, 5120)),
		dim3(256)>>>(
		momentum_w->devData, 
		wgrad->devData, 
		w->devData,
		momentum_b->devData, 
		bgrad->devData, 
		b->devData, 
		wgrad->getLen(),
		bgrad->getLen(),
		Config::instance()->getMomentum(), 
		Config::instance()->getLrate());
}

void SoftMax::clearMomentum()
{
	momentum_b->gpuClear();
	momentum_w->gpuClear();
}

void SoftMax::getCost(cuMatrix<double>*cost, int* y)
{
	g_getCost_1<<<dim3(1), dim3(256), sizeof(double) * 256>>>(outputs->devData, groudTruth->devData,
		cost->devData, y, outputs->rows, outputs->cols, batch);
	cudaDeviceSynchronize();
	getLastCudaError("g_getCost_1");

	g_getCost_2<<<dim3(1), dim3(256), sizeof(double) * 256>>>(cost->devData,  w->devData, lambda,
		w->getLen());
	cudaDeviceSynchronize();
	getLastCudaError("g_getCost_2");
}

cuMatrix<double>* SoftMax::getOutputs()
{
	return outputs;
}

cuMatrix<double>* SoftMax::getPreDelta()
{
	return preDelta;
}

cuMatrix<double>* SoftMax::getCurDelta()
{
	return curDelta;
}

void SoftMax::setPreDelta(cuMatrix<double>* _preDelta)
{
	preDelta = _preDelta;
}

void SoftMax::initRandom()
{
	srand(clock());
	double epsilon = 0.01f;
	for(int c = 0; c < w->channels; c++){
		for(int i = 0; i < w->rows; i++){
			for(int j=0; j < w->cols; j++){
				w->set(i,j, c, 1.0 * rand() / RAND_MAX *  2.0 * epsilon - epsilon);     
			}
		}
	}
	w->toGpu();
}
	
void SoftMax::initFromCheckpoint(FILE* file)
{
	double val = 0.0;
	for(int i = 0; i < w->rows; i++){
		for(int j=0; j< w->cols; j++){
			fscanf(file, "%lf", &val);
			w->set(i,j,0,val);
		}
	}
	
	for(int i = 0; i < b->rows; i++){
		for(int j = 0; j < b->cols; j++){
			fscanf(file, "%lf ", &val);
			b->set(i,j,0, val);
		}
	}
	w->toGpu();
	b->toGpu();
}

void SoftMax::save(FILE* file)
{
	w->toCpu();
	b->toCpu();
	for(int c = 0; c < w->channels; c++){
		for(int i = 0; i< w->rows; i++){
			for(int j=0; j< w->cols; j++){
				fprintf(file, "%lf ", w->get(i,j,c)); 
			}
		}
	}

	for(int c = 0; c < b->channels; c++){
		for(int i = 0; i < b->rows; i++){
			for(int j = 0; j < b->cols;  j++){
				fprintf(file, "%lf ", b->get(i,j,c));
			}
		}
	}
}

SoftMax::SoftMax(std::string name)
{
	m_name = name;
	ConfigFC* config = (ConfigFC*)Config::instance()->getLayerByName(m_name);
	LayerBase * preLayer = (LayerBase*)Layers::instance()->get(config->m_input);
	inputs = preLayer->getOutputs();
	batch = Config::instance()->getBatchSize();
	lambda = config->m_weightDecay;

	inputsize = inputs->cols * inputs->channels;
	outputsize = config->m_numFullConnectNeurons;

	NON_LINEARITY = Config::instance()->getNonLinearity();

	outputs = new cuMatrix<double>(batch, outputsize, 1);
	curDelta= new cuMatrix<double>(batch, outputsize, 1);
	preDelta= preLayer->getCurDelta();

	w     = new cuMatrix<double>(outputsize, inputsize, 1);
	wgrad = new cuMatrix<double>(outputsize, inputsize, 1);

	b     = new cuMatrix<double>(outputsize, 1, 1);
	bgrad = new cuMatrix<double>(outputsize, 1, 1);

	momentum_w = new cuMatrix<double>(outputsize, inputsize, 1);
	momentum_b = new cuMatrix<double>(outputsize, 1, 1);

	groudTruth = new cuMatrix<double>(batch, outputsize, 1);

	this->initRandom();
	Layers::instance()->set(m_name, this);
}