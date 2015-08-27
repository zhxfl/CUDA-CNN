#include "SoftMax.h"
#include "../common/cuBase.h"
#include "../common/cuMatrix.h"
#include "../common/Config.h"
#include "../layers/BranchLayer.h"
#include <math.h>

/*
* blocks : cuSoftMaxP->rows
* threads: cuSoftMaxP->cols
* shared : sizeof(float) * cuSoftMaxP->cols * 2
*/
__global__ void g_getSoftMaxP(float* softMaxP, float* b, int cols)
{
	int bid = blockIdx.x;
	extern __shared__ float _share[];
	float * _max = _share;
	float * _sum = _share + blockDim.x;
	float* sp = softMaxP + bid * cols;
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


__global__ void g_getSoftMaxDelta(float* softMaxDelta, float* softMaxP, float* groudTruth, int len)
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
__global__ void g_getSmrWgrad(float* wgrad, float* weight, float lambda, int len, int batch)
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
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_convert");

	matrixMulTB(inputs_format,
		w, outputs);

	int threads = std::min(512, outputs->cols);
	g_getSoftMaxP<<<outputs->rows, threads, sizeof(float) * threads * 2>>>(
		outputs->getDev(),
		b->getDev(), 
		outputs->cols);
	cudaStreamSynchronize(0);
	getLastCudaError("g_getSoftMaxP");
}

void SoftMax::backpropagation()
{
	g_getCost_1<<<dim3(1), dim3(256), sizeof(float) * 256>>>(outputs->getDev(), groudTruth->getDev(),
		cost->getDev(), predict, outputs->rows, outputs->cols, batch);
	cudaStreamSynchronize(0);
	getLastCudaError("g_getCost_1");

	g_getSoftMaxDelta<<<dim3(1), dim3(256)>>>(curDelta->getDev(),
		outputs->getDev(),
		groudTruth->getDev(), curDelta->getLen());
	cudaStreamSynchronize(0);
    getLastCudaError("g_getSoftMaxDelta");

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
	cudaStreamSynchronize(0);
	getLastCudaError("g_preDeltaFormat");
}

void SoftMax::getGrad()
{
    //printf("%d %d\n", inputs_format->rows, inputs_format->cols);
	matrixMulTA(curDelta, inputs_format, wgrad);

	g_getSmrWgrad<<<dim3(1), dim3(256)>>>(wgrad->getDev(),
		w->getDev(), lambda, wgrad->getLen(), batch);
	cudaStreamSynchronize(0);
	getLastCudaError("g_getSmrWgrad");

	if(curDelta->rows > MAX_THREADS)
	{
		printf("getSoftMaxDelta g_getBgrad > MAX_THREADS\n");
		exit(0);
	}
	g_getBgrad<<<dim3(curDelta->cols), dim3(curDelta->rows), 
		sizeof(float) * curDelta->rows>>>(
		curDelta->getDev(), 
		bgrad->getDev(),
		batch);
	cudaStreamSynchronize(0);
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
	g_getCost_2<<<dim3(1), dim3(256), sizeof(float) * 256>>>(cost->getDev(),  w->getDev(), lambda,
		w->getLen());
	cudaStreamSynchronize(0);
	getLastCudaError("g_getCost_2");
}

cuMatrix<float>* SoftMax::getOutputs()
{
	return outputs;
}

cuMatrix<float>* SoftMax::getCurDelta()
{
	return curDelta;
}

void SoftMax::setPreDelta(cuMatrix<float>* _preDelta)
{
	preDelta = _preDelta;
	preDelta_format = new cuMatrix<float>(preDelta->rows, preDelta->cols * preDelta->channels, 1);
}

void SoftMax::initRandom()
{
	//srand(clock());
	float initW = Config::instance()->getLayerByName(m_name)->m_initW;

	if(Config::instance()->getLayerByName(m_name)->isGaussian()){
		float epsilon = initW;
		for(int c = 0; c < w->channels; c++){
			float r1 = 0.01f + 5.0f * (rand()) / RAND_MAX;
			float r2 = 0.01f + 5.0f * (rand()) / RAND_MAX;
			createGaussian(w->getHost() + c * w->getArea(), r1,r2,
				w->rows, w->cols, w->channels,
				epsilon);
		}
	}
	else{
		for(int j = 0; j < w->getLen(); j++){
			w->getHost()[j] =  initW * (2.0f * rand() / RAND_MAX - 1.0f);
		}
	}
	w->toGpu();
}
	
void SoftMax::initFromCheckpoint(FILE* file)
{
	float val = 0.0;
	for(int i = 0; i < w->rows; i++){
		for(int j=0; j< w->cols; j++){
            if(fscanf(file, "%f", &val) == EOF){
                LOG("scanf fail", "result/log.txt");
            }
			w->set(i,j,0,val);
		}
	}
	
	for(int i = 0; i < b->rows; i++){
		for(int j = 0; j < b->cols; j++){
            if(fscanf(file, "%f ", &val) == EOF){
                LOG("scanf fail", "result/log.txt");
            }
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
				fprintf(file, "%f ", w->get(i,j,c)); 
			}
		}
	}

	for(int c = 0; c < b->channels; c++){
		for(int i = 0; i < b->rows; i++){
			for(int j = 0; j < b->cols;  j++){
				fprintf(file, "%f ", b->get(i,j,c));
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
		BranchLayer* bl = static_cast<BranchLayer*>(preLayer);
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

	inputs_format = new cuMatrix<float>(inputs->rows, inputs->cols * inputs->channels, 1);
	outputs = new cuMatrix<float>(batch, outputsize, 1);
	curDelta= new cuMatrix<float>(batch, outputsize, 1);
	this->setPreDelta(preDelta);

	w     = new cuMatrix<float>(outputsize, inputsize, 1);
	wgrad = new cuMatrix<float>(outputsize, inputsize, 1);

	b     = new cuMatrix<float>(outputsize, 1, 1);
	bgrad = new cuMatrix<float>(outputsize, 1, 1);

	momentum_w = new cuMatrix<float>(outputsize, inputsize, 1);
	momentum_b = new cuMatrix<float>(outputsize, 1, 1);

	groudTruth = new cuMatrix<float>(batch, outputsize, 1);

	this->initRandom();
	Layers::instance()->set(m_name, this);
}
