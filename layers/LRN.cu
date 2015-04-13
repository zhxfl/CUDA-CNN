#include "LRN.h"
#include <vector>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
#include "../common/Config.h"
#include "../common/cuBase.h"

#define USE_DOUBLE float
/*
* int _min = (outputDim * outputDim + 15) / 16 * 16;
* int remain = min(1024 / _min, outputAmount); //32
* int div = (outputAmount + remain - 1) / remain;//1
* dim3 block = dim3(batch, div);
* dim3 thread= dim3(min(outputDim * outputDim, _min), remain);
*/
__global__ void g_LRN_backpropagation(
	double* inputs,
	double* inputsDelta,
	double* outputsDelta,
	int inputDim,
	int outputDim,
	int InputArea,
	int outputArea,
	int batch,
	int kAmount,
	double lrn_k,
	int lrn_n, 
	double lrn_alpha,
	double lrn_belta);

/*
 * int _min = (outputDim * outputDim + 31) / 32 * 32;
 * int curDeltalen = curDelta->getLen();
 * int remain = min(1024 / _min, outputAmount); //32
 * int div = (outputAmount + remain - 1) / remain;//1
 * dim3 block = dim3(batch, div);
 * dim3 thread= dim3(min(outputDim * outputDim, _min), remain);
*/
__global__ void g_LRN_feedforward(
	double* inputs ,
	double* outputs,
	int inputDim,
	int outputDim,
	int InputArea,
	int outputArea,
	int batch,
	int kAmount,
	double lrn_k,
	int lrn_n, 
	double lrn_alpha,
	double lrn_belta);

void LRN::feedforward()
{
	int _min = (outputDim * outputDim + 31) / 32 * 32;

	if(_min < 256 && _min > 128) _min = 256;
	else if(_min < 128 && _min > 64) _min = 128;
	else if(_min < 64 && _min > 32) _min = 64;
	else if(_min < 32 && _min > 16) _min = 32;
	else _min = 16;

	int remain = min(1024 / _min, outputAmount); //32
	int div = (outputAmount + remain - 1) / remain;//1
	dim3 block = dim3(batch, div);
	dim3 thread= dim3(min(outputDim * outputDim, _min), remain);
	
	g_LRN_feedforward<<<block, thread>>>(
		inputs->getDev(),
		outputs->getDev(),
		inputDim,
		outputDim,
		inputs->getArea(),
		outputs->getArea(),
		batch,
		outputAmount,
		lrn_k,
		lrn_n,
		lrn_alpha,
		lrn_belta);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("LRN feedforward");

	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));
		g_nonLinearity<<<block, thread>>>(
			outputs->getDev(), 
			outputs->getLen(),
			NON_LINEARITY);
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("LRN::g_nonLinearity");
	}
	
}

void LRN::backpropagation()
{
	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));

		g_dnonLinearity<<<block, thread>>>(curDelta->getDev(),
			outputs->getDev(), curDelta->getLen(), NON_LINEARITY);

		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("ConvNCFM::g_dnonLinearity");
	}
	preDelta->gpuClear();

	int _min = (outputDim * outputDim + 31) / 32 * 32;
	if(_min < 256 && _min > 128) _min = 256;
	else if(_min < 128 && _min > 64) _min = 128;
	else if(_min < 64 && _min > 32) _min = 64;
	else if(_min < 32 && _min > 16) _min = 32;
	else _min = 16;

	int curDeltalen = curDelta->getLen();
	int remain = min(1024 / _min, outputAmount); //32
	int div = (outputAmount + remain - 1) / remain;//1
	dim3 block = dim3(batch, div);
	dim3 thread= dim3(min(outputDim * outputDim, _min), remain);

	g_LRN_backpropagation<<<block, thread>>>(inputs->getDev(),
		curDelta->getDev(),
		preDelta->getDev(),
		inputDim,
		outputDim,
		inputs->getArea(),
		outputs->getArea(),
		batch,
		outputAmount,
		lrn_k,
		lrn_n,
		lrn_alpha,
		lrn_belta);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("LRN backpropagation");
}

LRN::LRN(std::string name)
{	
	cost = NULL;
	m_name = name;
	ConfigLRN* config = (ConfigLRN*)Config::instance()->getLayerByName(m_name);
	ConvLayerBase * preLayer = (ConvLayerBase*)Layers::instance()->get(config->m_input);

	inputs = preLayer->getOutputs();
	inputDim = preLayer->outputDim;
	outputDim = inputDim;
	outputAmount = preLayer->outputAmount;
	inputAmount = outputAmount;
	
	NON_LINEARITY = config->m_nonLinearity;
	batch = Config::instance()->getBatchSize();
	
	/*local response nomarlization*/
	lrn_k = config->m_k;
	lrn_n = config->m_n;
	lrn_alpha = config->m_alpha;
	lrn_belta = config->m_belta;

	outputs  = new cuMatrix<double>(batch, outputDim * outputDim, outputAmount);
	curDelta = new cuMatrix<double>(batch, outputDim * outputDim, outputAmount);
	preDelta = preLayer->getCurDelta();

	Layers::instance()->set(m_name, this);
}

/*
 * int _min = (outputDim * outputDim + 31) / 32 * 32;
 * int curDeltalen = curDelta->getLen();
 * int remain = min(1024 / _min, outputAmount); //32
 * int div = (outputAmount + remain - 1) / remain;//1
 * dim3 block = dim3(batch, div);
 * dim3 thread= dim3(min(outputDim * outputDim, _min), remain);
*/
__global__ void g_LRN_feedforward(
	double* inputs ,
	double* outputs,
	int inputDim,
	int outputDim,
	int InputArea,
	int outputArea,
	int batch,
	int kAmount,
	double lrn_k,
	int lrn_n, 
	double lrn_alpha,
	double lrn_belta)
{
	int sp = blockIdx.x;
	//int k  = blockIdx.y;
	int k  = blockIdx.y * blockDim.y + threadIdx.y;
	if(k >= kAmount)return;

	int inputDim2   = inputDim  * inputDim;
	int outputDim2  = outputDim * outputDim;

	/*LRN*/
	for(int tidx = 0; tidx < outputDim2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < outputDim2)
		{
			int from, to;
			if(kAmount < lrn_n){
				from = 0;
				to = kAmount - 1;
			}else{
				int half = lrn_n >> 1;
				from = (k - half) >= 0 ? (k - half) : 0;
				to   = (k + half) <= (kAmount - 1) ? (k + half) : (kAmount - 1);
			}
			double u = 0.0;
			int offset = from * InputArea + sp * inputDim2 + idx;
			int koffset = InputArea * k + sp * inputDim2 + idx;
			double a = inputs[koffset];
			
			for(int j = from; j <= to; j++){
				double val = inputs[offset];
				u = u + val * val;
				offset += InputArea;
			}
			u = u * lrn_alpha / (to - from + 1) + lrn_k;
			//u = (double)pow((float)u, (float)lrn_belta);
			u = (double)pow((USE_DOUBLE)u, (USE_DOUBLE)lrn_belta);
			outputs[koffset] = a / u;
		}
	}
}

/*
 * int _min = (outputDim * outputDim + 31) / 32 * 32;
 * int curDeltalen = curDelta->getLen();
 * int remain = min(1024 / _min, outputAmount); //32
 * int div = (outputAmount + remain - 1) / remain;//1
 * dim3 block = dim3(batch, div);
 * dim3 thread= dim3(min(outputDim * outputDim, _min), remain);
*/

__global__ void g_LRN_backpropagation(
	double* inputs ,
	double* inputsDelta,
	double* outputsDelta,
	int inputDim,
	int outputDim,
	int InputArea,
	int outputArea,
	int batch,
	int kAmount,
	double lrn_k,
	int lrn_n, 
	double lrn_alpha,
	double lrn_belta)
{
	int sp = blockIdx.x;
	int k  = blockIdx.y * blockDim.y + threadIdx.y;
	if(k >= kAmount)return;

	int inputDim2  = inputDim  * inputDim;
	int outputDim2 = outputDim * outputDim;

	/*LRN*/
	for(int tidx = 0; tidx < outputDim2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < outputDim2)
		{
			int from, to;
			if(kAmount < lrn_n){
				from = 0;
				to = kAmount - 1;
			}else{
				int half = lrn_n >> 1;
				from = (k - half) >= 0 ? (k - half) : 0;
				to   = (k + half) <= (kAmount - 1) ? (k + half) : (kAmount - 1);
			}

			double u = 0.0;

			int offset = from * InputArea + sp * inputDim2 + idx;
			int koffset = InputArea * k + sp * inputDim2 + idx;

			double a = inputs[koffset];
			for(int j = from; j <= to; j++){
				double val = inputs[offset];
				u = u + val * val;
				offset += InputArea;
			}

			//
			u = u * lrn_alpha / (to - from + 1) + lrn_k;
			//double t1 = (double)pow((float)u, (float)(lrn_belta - 1)); //pow(u, lrn_belta - 1)
			double t1 = (double)pow((USE_DOUBLE)u, (USE_DOUBLE)(lrn_belta - 1)); //pow(u, lrn_belta - 1)
			double t2 = t1 * u;                //pow(u, lrn_belta)
			double t3 = t2 * t2;               //pow(u, 2.0 * lrn_belta)

			double u1 = t2 - 2.0 * lrn_belta * lrn_alpha * t1 * a * a / (to - from + 1);
			double u2 = t3;
			outputsDelta[koffset] = 
				inputsDelta[koffset] * u1 / u2;
		}
	}
}
