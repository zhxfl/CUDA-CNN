#include "FullConnect.h"
#include "../common/cuBase.h"
#include "../common/cuMatrix.h"
#include "../common/Config.h"
#include <math.h>

/*
* function: cuMatrix(batch, size, channel) to cuMatrix(batch, size * channel, 1)
* blocks  : dim3(batch)
* threads : dim3(min(512, cuPool[poolidx]->cols))
*/
__global__ void g_convert(double* cuPool, 
	double*cuPoolToFlActi,
	int batch,
	int size,
	int channel);

__global__ void g_FullConnectDropW(double * w, 
	double * dropW,
	double* afterDropW,
	int len);

__global__ void g_FullConnectFeedforward(double* acti, 
	double* b,
	int NumofNeurons, 
	int NONLIN);

__global__ void g_FullConnectActi(double* acti,
	double* b,
	int NumofNeurons, 
	int NONLIN);

__global__ void g_FullConnectWgrad(double* wgrad, 
	double* w, 
	double* dropM,
	int len, 
	double lambda, 
	int batch);

__global__ void g_preDeltaFormat(double* cuPoolFlDelta, 
	double* cuPoolDelta, int batch, int size, int channels);



__global__ void g_FullConnectActi(double* acti, double* b, int NumofNeurons, int NONLIN)
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

__global__ void g_FullConnectWgrad(double* wgrad, double* w, double* dropM, int len, double lambda, int batch)
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


/*
* blocks  : cuFullConnectActi[hl]->rows;
* threads : dim3(min(512, len));
*/
__global__ void g_FullConnectFeedforward(double* acti, double* b, int NumofNeurons, int NONLIN)
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

__global__ void g_FullConnectDropW(double * w, double * dropW, double* afterDropW, int len)
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
function: g_preDeltaFormat
threads : <<<dim3(batch), dim3(512)>>> 
*/
__global__ void g_preDeltaFormat(double* cuPoolFlDelta, 
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


void FullConnect::feedforward()
{
	//drop
	dim3 block  = inputs->rows;
	dim3 thread = min(512, inputs->cols);

	//convert 
	g_convert<<<block, thread>>>(
		inputs->devData, 
		inputs_format->devData, 
		inputs->rows, 
		inputs->cols,
		inputs->channels);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("g_convert");

	//drop w
	thread = min(512, w->getLen());
	block  = min(512, (w->getLen() + thread.x - 1) / thread.x);
	g_FullConnectDropW<<<block, thread>>>(w->devData,
		dropW->devData, afterDropW->devData,
		afterDropW->getLen());

	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("g_FullConnectDropW");

 	matrixMulTB(inputs_format, afterDropW,
 		outputs);

	thread = min(512, outputs->cols);
	block  = outputs->rows;
	g_FullConnectActi<<<block, thread>>>(outputs->devData,
		b->devData,
		outputs->cols,
		NON_LINEARITY);

	cudaDeviceSynchronize();
	getLastCudaError("g_FullConnectActi");
}


void FullConnect::getCost(cuMatrix<double>*cost)
{
	if(fabs(lambda) >= 1e-10)
	{
		g_getCost_2<<<dim3(1), dim3(256), sizeof(double) * 256>>>(cost->devData,
			w->devData,
			lambda,
			w->getLen());
		cudaDeviceSynchronize();
		getLastCudaError("g_getCost_2");
	}
}

FullConnect::FullConnect(cuMatrix<double>* _inputs,
	int _batch,
	double _lambda,	
	int _neurons,
	double _dropRate,
	int _NON_LINEARITY)
{
	inputs = _inputs;
	batch = _batch;
	lambda = _lambda;
	inputsize = inputs->cols * inputs->channels;
	outputsize = _neurons;
	dropRate = _dropRate;

	NON_LINEARITY = _NON_LINEARITY;

	inputs_format = new cuMatrix<double>(inputs->rows, inputs->cols * inputs->channels, 1);
	outputs       = new cuMatrix<double>(batch, outputsize, 1);
	curDelta      = new cuMatrix<double>(batch, outputsize, 1);


	w          = new cuMatrix<double>(outputsize, inputsize, 1);
	wgrad      = new cuMatrix<double>(outputsize, inputsize, 1);
	dropW      = new cuMatrix<double>(outputsize, inputsize, 1);
	afterDropW = new cuMatrix<double>(outputsize, inputsize, 1);
	
	b     = new cuMatrix<double>(outputsize, 1, 1);
	bgrad = new cuMatrix<double>(outputsize, 1, 1);

	momentum_w = new cuMatrix<double>(outputsize, inputsize, 1);
	momentum_b = new cuMatrix<double>(outputsize, 1, 1);

	dropDelta(dropW, dropRate);
}

void FullConnect::drop()
{
	//if(fabs(dropRate) >= 0)
	dropDelta(dropW, dropRate);
}

void FullConnect::drop(double rate)
{
	//if(fabs(dropRate) >= 0)
	dropDelta(dropW, rate);
}



void FullConnect::backpropagation()
{
	if(NON_LINEARITY >= 0){
		g_dnonLinearity<<<dim3(256), dim3(256)>>>(curDelta->devData,
			outputs->devData, outputs->getLen(), NON_LINEARITY);
		cudaDeviceSynchronize();
		getLastCudaError("g_dnonLinearity");
	}

	//preDelta 
	matrixMul(curDelta, afterDropW, preDelta_format);
	dim3 block = batch;
	dim3 thread= min(512, preDelta->channels * preDelta->cols);
	g_preDeltaFormat<<<block, thread>>>(
		preDelta_format->devData,
		preDelta->devData,
		preDelta->rows,
		preDelta->cols,
		preDelta->channels);
	cudaDeviceSynchronize();
	getLastCudaError("g_preDeltaFormat");
}

void FullConnect::getGrad()
{
	matrixMulTA(curDelta,
		inputs,
		wgrad);

	g_FullConnectWgrad<<<dim3(256), dim3(256)>>>(wgrad->devData,
		w->devData,
		dropW->devData,
		wgrad->getLen(),
		lambda,
		batch);
	cudaDeviceSynchronize();
	getLastCudaError("g_FullConnectWgrad");


	if(curDelta->rows > MAX_THREADS)
	{
		printf("getFullConnectDelta g_getBgrad > MAX_THREADS\n");
		exit(0);
	}
	g_getBgrad<<<dim3(curDelta->cols), dim3(curDelta->rows),
		sizeof(double) * curDelta->rows>>>
		(curDelta->devData, bgrad->devData, batch);
	cudaDeviceSynchronize();
}

void FullConnect::updateWeight()
{
	dim3 block = min((momentum_w->getLen() + 255) / 256, 5120);
	dim3 thread= 256;

	g_vecAdd<<<block, thread>>>(momentum_w->devData, wgrad->devData, w->devData,
		momentum_b->devData, bgrad->devData, b->devData,
		wgrad->getLen(), bgrad->getLen(), 
		Config::instance()->getMomentum(),
		Config::instance()->getLrate());
}

void FullConnect::clearMomentum()
{
	momentum_b->gpuClear();
	momentum_w->gpuClear();
}

cuMatrix<double>* FullConnect::getOutputs()
{
	return outputs;
}

cuMatrix<double>* FullConnect::getPreDelta()
{
	return preDelta;
}

cuMatrix<double>* FullConnect::getCurDelta()
{
	return curDelta;
}

void FullConnect::setPreDelta(cuMatrix<double>* _preDelta)
{
	preDelta = _preDelta;
	preDelta_format = new cuMatrix<double>(preDelta->rows, preDelta->cols * preDelta->channels, 1);
}

/*
* function: cuMatrix(batch, size, channel) to cuMatrix(batch, size * channel, 1)
* blocks  : dim3(batch)
* threads : dim3(min(512, cuPool[poolidx]->cols))
*/
__global__ void g_convert(double* cuPool, double*cuPoolToFlActi, int batch, int size, int channel)
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

void FullConnect::convert()
{
	int threads = min(512, inputs->cols);
	g_convert<<<dim3(inputs->rows), threads>>>
		(inputs->devData, 
		inputs_format->devData, 
		inputs->rows,
		inputs->cols,
		inputs->channels);
	cudaDeviceSynchronize();
	getLastCudaError("convert");
}

void FullConnect::initRandom()
{
	srand(clock());
 	double epsilon = sqrt((double)6) / sqrt((double)(inputsize + outputsize + 1));

 	for(int c=0; c < w->channels; c++){
 		for(int i=0; i < w->rows; i++){
 			for(int j=0; j< w->cols; j++){
 				w->set(i,j, c, 1.0 * rand() / RAND_MAX *  2.0 * epsilon - epsilon);
 			}
 		}
 	}
 	w->toGpu();
}

void FullConnect::initFromCheckpoint(FILE* file)
{
	double val = 0.0;
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

void FullConnect::save(FILE* file)
{
	w->toCpu();
	b->toCpu();
	for(int c = 0; c < w->channels; c++){
		for(int i = 0; i < w->rows; i++){
			for(int j = 0; j < w->cols; j++){
				fprintf(file, "%lf ", w->get(i,j,c));
			}
		}
	}

	for(int c = 0; c < b->channels; c++){
		for(int i = 0; i < b->rows; i++){
			for(int j = 0; j < b->cols;  j++){
				fprintf(file, "%lf ", b->get(i,j, c));
			}
		}
	}
}

