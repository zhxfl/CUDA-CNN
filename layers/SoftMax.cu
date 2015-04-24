#include "SoftMax.h"
#include "../common/cuBase.h"
#include "../common/cuMatrix.h"
#include "../common/Config.h"
#include "../layers/BrachLayer.h"
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
			wgrad[id] = lambda * weight[id] + wgrad[id] / batch;
		}
	}
}

void SoftMax::feedforward()
{
	dim3 block  = inputs->rows;
	dim3 thread = min(512, inputs->cols);

	//convert 
	g_convert<<<block, thread>>>(
		inputs->getDev(), 
		inputs_format->getDev(), 
		inputs->rows, 
		inputs->cols,
		inputs->channels);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("g_convert");

	matrixMulTB(inputs_format,
		w, outputs);

	int threads = std::min(512, outputs->cols);
	g_getSoftMaxP<<<outputs->rows, threads, sizeof(double) * threads * 2>>>(
		outputs->getDev(),
		b->getDev(), 
		outputs->cols);
	cudaDeviceSynchronize();
	getLastCudaError("g_getSoftMaxP");
}

void SoftMax::backpropagation()
{
	g_getCost_1<<<dim3(1), dim3(256), sizeof(double) * 256>>>(outputs->getDev(), groudTruth->getDev(),
		cost->getDev(), predict, outputs->rows, outputs->cols, batch);
	cudaDeviceSynchronize();
	getLastCudaError("g_getCost_1");

	g_getSoftMaxDelta<<<dim3(1), dim3(256)>>>(curDelta->getDev(),
		outputs->getDev(),
		groudTruth->getDev(), curDelta->getLen());
	cudaDeviceSynchronize();

	matrixMul(curDelta,
		w, preDelta_format);

	dim3 block = batch;
	dim3 thread= min(512, preDelta->channels * preDelta->cols);
	g_preDeltaFormat<<<block, thread>>>(
		preDelta_format->getDev(),
		preDelta->getDev(),
		preDelta->rows,
		preDelta->cols,
		preDelta->channels);
	cudaDeviceSynchronize();
	getLastCudaError("g_preDeltaFormat");
}

void SoftMax::getGrad()
{
	matrixMulTA(curDelta, inputs_format, wgrad);

	g_getSmrWgrad<<<dim3(1), dim3(256)>>>(wgrad->getDev(),
		w->getDev(), lambda, wgrad->getLen(), batch);
	cudaDeviceSynchronize();

	if(curDelta->rows > MAX_THREADS)
	{
		printf("getSoftMaxDelta g_getBgrad > MAX_THREADS\n");
		exit(0);
	}
	g_getBgrad<<<dim3(curDelta->cols), dim3(curDelta->rows), 
		sizeof(double) * curDelta->rows>>>(
		curDelta->getDev(), 
		bgrad->getDev(),
		batch);
	cudaDeviceSynchronize();
	getLastCudaError("g_getBgrad");
}

void SoftMax::updateWeight()
{
	g_vecAdd<<<dim3(min((momentum_w->getLen() + 255) / 256, 5120)),
		dim3(256)>>>(
		momentum_w->getDev(), 
		wgrad->getDev(), 
		w->getDev(),
		momentum_b->getDev(), 
		bgrad->getDev(), 
		b->getDev(), 
		wgrad->getLen(),
		bgrad->getLen(),
		Config::instance()->getMomentum(), 
		Config::instance()->getLrate(), Config::instance()->getLrate());
}

void SoftMax::clearMomentum()
{
	momentum_b->gpuClear();
	momentum_w->gpuClear();
}

void SoftMax::calCost()
{
	g_getCost_2<<<dim3(1), dim3(256), sizeof(double) * 256>>>(cost->getDev(),  w->getDev(), lambda,
		w->getLen());
	cudaDeviceSynchronize();
	getLastCudaError("g_getCost_2");
}

cuMatrix<double>* SoftMax::getOutputs()
{
	return outputs;
}

cuMatrix<double>* SoftMax::getCurDelta()
{
	return curDelta;
}

void SoftMax::setPreDelta(cuMatrix<double>* _preDelta)
{
	preDelta = _preDelta;
	preDelta_format = new cuMatrix<double>(preDelta->rows, preDelta->cols * preDelta->channels, 1);
}

void SoftMax::initRandom()
{
	//srand(clock());
	double initW = Config::instance()->getLayerByName(m_name)->m_initW;

	if(Config::instance()->getLayerByName(m_name)->isGaussian()){
		double epsilon = initW;
		for(int c = 0; c < w->channels; c++){
			double r1 = 0.01 + 5 * (rand()) / RAND_MAX;
			double r2 = 0.01 + 5 * (rand()) / RAND_MAX;
			createGaussian(w->getHost() + c * w->getArea(), r1,r2,
				w->rows, w->cols, w->channels,
				epsilon);
		}
	}
	else{
		for(int j = 0; j < w->getLen(); j++){
			w->getHost()[j] =  initW * (2.0 * rand() / RAND_MAX - 1.0);
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
	if(inputs == NULL){
		/*inputs = NULL the type must be BranchLayers*/
		Assert(Config::instance()->getLayerByName(config->m_input)->isBranchLayer());
		Assert(config->m_subInput != std::string("NULL"));
		BrachLayer* bl = static_cast<BrachLayer*>(preLayer);
		inputs = bl->getSubOutput(config->m_subInput);
		preDelta = bl->getSubCurDelta(config->m_subInput);
	}else{
		preDelta = preLayer->getCurDelta();
	}

	batch = Config::instance()->getBatchSize();
	lambda = config->m_weightDecay;

	inputsize = inputs->cols * inputs->channels;
	outputsize = config->m_numFullConnectNeurons;

	NON_LINEARITY = config->m_nonLinearity;

	inputs_format = new cuMatrix<double>(inputs->rows, inputs->cols * inputs->channels, 1);
	outputs = new cuMatrix<double>(batch, outputsize, 1);
	curDelta= new cuMatrix<double>(batch, outputsize, 1);
	this->setPreDelta(preDelta);

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