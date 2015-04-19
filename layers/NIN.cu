#include "NIN.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
#include "../layers/BrachLayer.h"


/*
 * dim3 thread = min(512, inputs->cols);
 * dim3 block  = dim3(inputs->rows, inputs->channels);
*/
__global__ void g_NIN_backpropagation(
	double* _curDelta,
	double* _w,
	double* _nextDelta,
	int rows, int cols, int channels);

/*
 * block = dim3(channels, cols);
 * thread= dim3(rows);
 * wgradTmp   = new cuMatrix<double>(channel, cols, batch)
 * w = new cuMatrix<double>(channel, cols, 1)
*/

__global__ void g_NIN_wgrad_Add(
	double* _WgradTmp,
	double* Wgrad,
	double* w,
	int rows,
	int cols,
	int channels,
	double lambda);

/*
 * dim3 block = dim3(rows, channel);
 * dim3 thread= dim3(min(cols, 512));
*/
__global__ void g_NIN_wgrad(
	double* _inputs,
	double* _curDelta,
	double* _wgradTmp,
	int rows,
	int cols,
	int channels);

/*
 * function: get convolution layer and pooling output
 * dim3 block = dim3(batch, amount);
 * dim3 thread= dim3(min(outputDim * outputDim, 512));
 * const kernelsize = 1
*/

__global__ void g_NIN_feedforward(
	double* _inputs,
	double* _w,
	double* _b,
	double* _outputs,
	int rows,
	int cols, 
	int channels);

/*
 * block = dim3(channel, cols);
 * thread= dim3(rows); 
*/
__global__ void g_NIN_Bgrad(double* _delta,
	double* bgrad,
	int rows,
	int cols,
	int channels);


void NIN::calCost()
{
	cost->gpuClear();
	g_getCost_2<<<dim3(1), dim3(256), sizeof(double) * 256>>>(cost->getDev(), 
		w->getDev(), 
		lambda,
		w->getLen());
	cudaDeviceSynchronize();
	getLastCudaError("NIN:g_getCost_2");
}

void NIN::feedforward()
{
	dim3 block  = dim3(inputs->rows, inputs->channels);
	dim3 thread = min(512, inputs->cols);

	g_NIN_feedforward<<<block, thread>>>(
		inputs->getDev(),
		w->getDev(),
		b->getDev(),
		outputs->getDev(),
		inputs->rows, 
		inputs->cols,
		inputs->channels);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("NIN:g_NIN_feedforward");
}

void NIN::backpropagation()
{
	dim3 block  = dim3(inputs->rows, inputs->channels);
	dim3 thread = min(512, inputs->cols);

	//preDelta->gpuClear();

	g_NIN_backpropagation<<<block, thread>>>(
		curDelta->getDev(),
		w->getDev(),
		preDelta->getDev(),
		inputs->rows, inputs->cols, inputs->channels);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("NIN::g_NIN_backpropagation");

// 	curDelta->toCpu();
// 	preDelta->toCpu();
// 	for(int i = 0; i < curDelta->getLen(); i++){
// 		if(curDelta->getHost()[i] != preDelta->getHost()[i]){
// 			sprintf(logStr, "%d %lf %lf\n", i, curDelta->getHost()[i], preDelta->getHost()[i]);
// 		}
// 	}
// 	exit(0);
}

/*
 * block    = dim3(channels, cols);
 * thread   = dim3(rows);
 * wgradTmp = new cuMatrix<double>(rows, channel, cols)
 * w        = new cuMatrix<double>(channel, cols, 1)
*/

__global__ void g_NIN_wgrad_Add(
	double* _WgradTmp,
	double* Wgrad,
	double* w,
	int rows,
	int cols,
	int channels,
	double lambda)
{
	extern __shared__ double _sum[];
	int channel = blockIdx.x;
	int col     = blockIdx.y;
	int tid     = threadIdx.x;
	_sum[tid] = 0;
	__syncthreads();


	for(int i = 0; i < rows; i += blockDim.x){
		int row = i + threadIdx.x;
		if(row < rows){
			_sum[threadIdx.x] += _WgradTmp[channel * rows * cols + row * cols + col];
		}
	}
	__syncthreads();

	int len = rows;
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
		Wgrad[channel * cols + col] = _sum[0] / rows + w[channel * cols + col] * lambda;
	}
}


void NIN::getGrad()
{
	int rows    = inputs->rows;
	int cols    = inputs->cols;
	int channel = inputs->channels;

	dim3 block = dim3(rows, channel);
	dim3 thread= dim3(min(cols, 512));

	g_NIN_wgrad<<<block, thread>>>(
		inputs->getDev(),
		curDelta->getDev(),
		wgradTmp->getDev(),
		rows,
		cols,
		channel);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("NIN::g_NIN_wgrad");

	block = dim3(channel, cols);
	thread= dim3(rows); 

	g_NIN_wgrad_Add<<<block, thread, sizeof(double) * rows>>>(
		wgradTmp->getDev(),
		wgrad->getDev(),
		w->getDev(),
		rows,
		cols,
		channel,
		lambda);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("NIN::g_NIN_wgrad_Add");

	block = dim3(channel, cols);
	thread= dim3(rows); 

	g_NIN_Bgrad<<<block, thread, sizeof(double) * rows>>>
		(curDelta->getDev(),
		bgrad->getDev(),
		rows,
		cols,
		channel);

	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("NIN::g_NIN_Bgrad");
}

void NIN::updateWeight()
{
// 	int len = (w->getLen()) / 2;
// 	dim3 thread = min(512, len);
// 	dim3 block  = min((len + thread.x - 1) / thread.x, 5120);
	dim3 block = min((momentum_w->getLen() + 255) / 256, 5120);
	dim3 thread= 256;
	g_vecAdd<<<block, thread>>>(momentum_w->getDev(), wgrad->getDev(), w->getDev(),
		momentum_b->getDev(), bgrad->getDev(), b->getDev(),
		w->getLen(), b->getLen(), 
		Config::instance()->getMomentum(),
		Config::instance()->getLrate());
}

NIN::NIN(std::string name)
{
	m_name = name;
	ConfigNIN *config = (ConfigNIN *) Config::instance()->getLayerByName(m_name);
	ConvLayerBase* preLayer = (ConvLayerBase*) Layers::instance()->get(config->m_input);

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

	inputDim = preLayer->outputDim;
	outputDim= inputDim;
	inputAmount = preLayer->outputAmount;
	outputAmount = inputAmount;
	lambda = config->m_weightDecay;

	int rows    = inputs->rows;
	int cols    = inputs->cols;
	int channel = inputs->channels;

	assert(rows == Config::instance()->getBatchSize());

	outputs    = new cuMatrix<double>(rows, cols, channel);
	curDelta   = new cuMatrix<double>(rows, cols, channel);

	w          = new cuMatrix<double>(channel, cols, 1);
	b          = new cuMatrix<double>(channel, cols, 1);
	wgrad      = new cuMatrix<double>(channel, cols, 1);
	bgrad      = new cuMatrix<double>(channel, cols, 1);
	wgradTmp   = new cuMatrix<double>(rows, cols, channel);
	momentum_w = new cuMatrix<double>(channel, cols, 1);
	momentum_b = new cuMatrix<double>(channel, cols, 1);

	this->initRandom();
	Layers::instance()->set(m_name, this);
}

void NIN::save(FILE* file)
{
	w->toCpu();
	for(int c = 0; c < w->channels; c++){
		for(int i = 0; i < w->rows; i++){
			for(int j = 0; j < w->cols; j++){
				fprintf(file, "%lf ", w->get(i, j, c));
			}
		}
	}
	b->toCpu();
	for(int c = 0; c < b->channels; c++){
		for(int i = 0; i < b->rows; i++){
			for(int j = 0; j < b->cols; j++){
				fprintf(file, "%lf ", b->get(i, j, c));
			}
		}
	}
}

void NIN::clearMomentum()
{
	momentum_b->gpuClear();
	momentum_w->gpuClear();
}

void NIN::initRandom()
{
	for(int i = 0; i < w->getLen(); i++){
		w->getHost()[i] =  1.0;
	}
	w->toGpu();
}

void NIN::initFromCheckpoint(FILE* file)
{
	double val = 0;
	for(int c = 0; c < w->channels; c++){
		for(int i = 0; i < w->rows; i++){
			for(int j = 0; j < w->cols; j++){
				fscanf(file, "%lf", &val);
				w->set(i, j, c, val);
			}
		}
	}


	for(int c = 0; c < b->channels; c++){
		for(int i = 0; i < b->rows; i++){
			for(int j = 0; j < b->cols; j++){
				fscanf(file, "%lf", &val);
				b->set(i, j, c, val);
			}
		}
	}
	w->toGpu();
	b->toGpu();
}


/*
 * dim3 block  = dim3(inputs->rows, inputs->channels);
 * dim3 thread = min(512, inputs->cols);
*/

__global__ void g_NIN_feedforward(
	double* _inputs,
	double* _w,
	double* _b,
	double* _outputs,
	int rows,
	int cols, 
	int channels)
{
	int row     = blockIdx.x;
	int channel = blockIdx.y;
	
	int skip = channel * rows * cols + row * cols;
	double* inputs = _inputs + skip;
	double* outputs= _outputs+ skip;
// 	if(threadIdx.x == 0)
// 		sprintf(logStr, "block(%d %d) skip = %d\n", blockIdx.x, blockIdx.y, skip);
	double* w = _w + channel * cols;
	double* b = _b + channel * cols;

	for(int i = 0; i < cols; i += blockDim.x){
		int id = i + threadIdx.x;
		if(id < cols){
			outputs[id] = inputs[id] * w[id] + b[id];
		}
	}
}

/*
 * dim3 block  = dim3(inputs->rows, inputs->channels);
 * dim3 thread = min(512, inputs->cols);
*/
__global__ void g_NIN_backpropagation(
	double* _curDelta,
	double* _w,
	double* _nextDelta,
	int rows, int cols, int channels)
{
	int row     = blockIdx.x;
	int channel = blockIdx.y;

	int skip = channel * rows * cols + row * cols;
	double* curDelta = _curDelta + skip;
	double* nextDelta= _nextDelta+ skip;

	double* w = _w + channel * cols;

	for(int i = 0; i < cols; i += blockDim.x){
		int id = i + threadIdx.x;
		if(id < cols){
			nextDelta[id] = curDelta[id] * w[id];
		}
	}
}

/*
 * dim3 block = dim3(rows, channel);
 * dim3 thread= dim3(min(cols, 512));
 * wgradTmp   = new cuMatrix<double>(rows, cols, channel);
*/
__global__ void g_NIN_wgrad(
	double* _inputs,
	double* _curDelta,
	double* _wgradTmp,
	int rows,
	int cols,
	int channels)
{
	int row     = blockIdx.x;
	int channel = blockIdx.y;

	int skip = channel * rows * cols + row * cols;
	double* inputs   = _inputs   + skip;
	double* curDelta = _curDelta + skip;
	double* wgradTmp = _wgradTmp + skip;

	for(int i = 0; i < cols; i += blockDim.x){
		int id = i + threadIdx.x;
		if(id < cols){
			wgradTmp[id] = inputs[id] * curDelta[id];
		}
	}
}


/*
* block = dim3(channel, cols);
* thread= dim3(rows); 
*/
__global__ void g_NIN_Bgrad(double* _delta,
	double* bgrad,
	int rows,
	int cols,
	int channels)
{
	extern __shared__ double _sum[];
	int channel = blockIdx.x;
	int col     = blockIdx.y;
	int row     = threadIdx.x;
	double delta = _delta[channel * rows * cols + row * cols + col];
	_sum[row] = delta;
	__syncthreads();

	int len = rows;
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
		bgrad[channel * cols + col] = _sum[0] / rows;
	}
}