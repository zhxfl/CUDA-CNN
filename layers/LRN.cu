#include "LRN.h"
#include <vector>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
#include "../common/Config.h"
#include "../common/cuBase.h"

/*
* function: unLRN
*/
__global__ void g_backpropagation(
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

__global__ void g_feedforward(
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
	dim3 block = dim3(batch, amount, Config::instance()->getChannels());
	dim3 thread= dim3(min(inputDim * inputDim, 512));
	
	g_feedforward<<<block, thread>>>(
		inputs->getDev(),
		outputs->getDev(),
		inputDim,
		outputDim,
		inputs->getArea(),
		outputs->getArea(),
		batch,
		amount,
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

	int curDeltalen = curDelta->getLen();
	dim3 block = dim3(batch, amount, Config::instance()->getChannels());
	dim3 thread= dim3(min(inputDim * inputDim, 512));

	g_backpropagation<<<block, thread>>>(inputs->getDev(),
		curDelta->getDev(),
		preDelta->getDev(),
		inputDim,
		outputDim,
		inputs->getArea(),
		outputs->getArea(),
		batch,
		amount,
		lrn_k,
		lrn_n,
		lrn_alpha,
		lrn_belta);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("LRN backpropagation");
}

LRN::LRN(std::string name)
{	
	m_name = name;
	ConfigLRN* config = (ConfigLRN*)Config::instance()->getLayerByName(m_name);
	ConvLayerBase * preLayer = (ConvLayerBase*)Layers::instance()->get(config->m_input);

	inputs = preLayer->getOutputs();
	inputDim = preLayer->outputDim;
	outputDim = inputDim;
	amount = preLayer->outputAmount;
	inputAmount = amount;
	outputAmount = amount;
	
	NON_LINEARITY = config->m_nonLinearity;
	batch = Config::instance()->getBatchSize();
	
	/*local response nomarlization*/
	lrn_k = config->m_k;
	lrn_n = config->m_n;
	lrn_alpha = config->m_alpha;
	lrn_belta = config->m_belta;

	int channels = inputs->channels;

	outputs  = new cuMatrix<double>(batch, amount * outputDim * outputDim, channels);
	curDelta = new cuMatrix<double>(batch, amount * outputDim * outputDim, channels);
	preDelta = preLayer->getCurDelta();

	Layers::instance()->set(m_name, this);
}

/*
*blocks : dim3(batch, cuKernelScan[0], Config::instance()->getChannels()),
*threads: dim3(min(convOutputSize * convOutputSize, 512));
*/

__global__ void g_feedforward(
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
	int k  = blockIdx.y;
	int c  = blockIdx.z;

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
				from = (k - lrn_n / 2) >= 0 ? (k - lrn_n / 2) : 0;
				to   = (k + lrn_n / 2) <= (kAmount - 1) ? (k + lrn_n / 2) : (kAmount - 1);
			}
			double u = 0.0;
			int skip = InputArea * c + (sp * kAmount) * inputDim2;
			double a = inputs[skip + inputDim2 * k + idx];
			for(int j = from; j <= to; j++){
				double val = inputs[skip + j * inputDim2 + idx];
				u = u + val * val;
			}
			u = u * lrn_alpha + lrn_k;
			u = pow(u, lrn_belta);
			outputs[skip + outputDim2 * k + idx] = a / u;
		}
	}
}

/*
* function: unLRN
*/
__global__ void g_backpropagation(
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
	int k  = blockIdx.y;
	int c  = blockIdx.z;

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
				from = (k - lrn_n / 2) >= 0 ? (k - lrn_n / 2) : 0;
				to   = (k + lrn_n / 2) <= (kAmount - 1) ? (k + lrn_n / 2) : (kAmount - 1);
			}

			double u = 0.0;
			int skip = InputArea * c + (sp * kAmount) * inputDim2;
			double a = inputs[skip + k * inputDim2 + idx];
			for(int j = from; j <= to; j++){
				double val = inputs[skip + j * inputDim2 + idx];
				u = u + val * val;
			}
			u = u * lrn_alpha + lrn_k;
			double u1 = pow(u, lrn_belta) - 2.0 * lrn_belta * lrn_alpha * pow(u, lrn_belta - 1) * a * a;
			double u2 = pow(u, 2.0 * lrn_belta);
			outputsDelta[skip + k * outputDim2 + idx] = 
				inputsDelta[skip + k * inputDim2 + idx] * u1 / u2;
		}
	}
}
