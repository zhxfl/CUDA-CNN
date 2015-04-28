#include "CombineLayer.h"
#include <vector>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
#include "../common/Config.h"
#include "../common/cuBase.h"


/*
 * dim3 block = dim3(batch, inputsSkip->getLen());
 * dim3 thread= dim3(min(preDelta[0]->getLen() / batch, 1024));
*/

__global__ void g_CombineLayer_backpropagation(
	float** preDeltas,
	float* curDeltas,
	int* skip,
	int* cols,
	int* channels,
	int batch,
	int curDeltaCols);

/* 
 * dim3 block = dim3(batch, skip->getLen());
 * dim3 thread= dim3(min(outputs->getLen() / batch, 1024));
*/

__global__ void g_CombineLayer_feedforward(
	float** inputs,
	float* outputs,
	int* skip,
	int* cols,
	int* channels,
	int batch,
	int outputCols);


void CombineLayer::feedforward()
{
	/*spread multi-inputs to output*/
	dim3 block = dim3(batch, inputsSkip->getLen());
	dim3 thread= dim3(min(outputs->getLen() / batch, 1024));
	
	g_CombineLayer_feedforward<<<block, thread>>>(
		inputs.m_devPoint,
		outputs->getDev(),
		inputsSkip->getDev(),
		inputsCols->getDev(),
		inputsChannels->getDev(),
		batch,
		outputs->cols);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("CombineLayer feedforward");

	//inputsChannels->toCpu();

#ifdef CombineLayer_feedforward_Checking 
	outputs->toCpu();
	for(int i = 0; i < inputs.size(); i++){
		inputs[i]->toCpu();
		for(int j = 0; j < inputs[i]->getLen(); j++){
			printf("%f ", inputs[i]->getHost()[j]);
		}printf("\n");
	}
	printf("\n\noutputs\n\n");
	for(int i = 0; i < outputs->getLen(); i++){
		printf("%f ", outputs->getHost()[i]);
	}printf("\n");
#endif
	
}

void CombineLayer::backpropagation()
{
	/*copy curDelta to multi-preDelta*/
	dim3 block = dim3(batch, inputsSkip->getLen());
	dim3 thread= dim3(min(preDelta[0]->getLen() / batch, 1024));

	g_CombineLayer_backpropagation<<<block, thread>>>(
		preDelta.m_devPoint,
		curDelta->getDev(),
		inputsSkip->getDev(),
		inputsCols->getDev(),
		inputsChannels->getDev(),
		batch,
		curDelta->cols);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("combineLayer backpropagation");

#ifdef CombineLayer_backpropagation_checking
	curDelta->toCpu();
	for(int i = 0; i < inputs.size(); i++){
		preDelta[i]->toCpu();
		for(int j = 0; j < preDelta[i]->getLen(); j++){
			printf("%f ", preDelta[i]->getHost()[j]);
		}printf("\n");
	}
	printf("\n\noutputs\n\n");
	for(int i = 0; i < curDelta->getLen(); i++){
		printf("%f ", curDelta->getHost()[i]);
	}printf("\n");

	exit(0);
#endif
}

CombineLayer::CombineLayer(std::string name)
{	
	cost = NULL;
	m_name = name;
	ConfigCombineLayer* config = (ConfigCombineLayer*)Config::instance()->getLayerByName(m_name);
	
	Assert(config->m_input == std::string("NULL"));
	
	/*multi-inputs*/
	/*suppose the input certainly not the BranLayers's sub-output*/

	inputsSkip     = new cuMatrix<int>(config->m_inputs.size(), 1, 1);
	inputsChannels = new cuMatrix<int>(config->m_inputs.size(), 1, 1);
	inputsCols     = new cuMatrix<int>(config->m_inputs.size(), 1, 1);

	int len = 0;
	for(int i = 0; i < config->m_inputs.size(); i++){
		ConvLayerBase * preLayer = (ConvLayerBase*)Layers::instance()->get(config->m_inputs[i]);
		inputs.push_back(preLayer->getOutputs());
		preDelta.push_back(preLayer->getCurDelta());

		inputsSkip->set(i, 0, 0, len);
		int area = preLayer->getOutputs()->cols * preLayer->getOutputs()->channels;

		inputsCols->set(i, 0, 0, preLayer->getOutputs()->cols);
		inputsChannels->set(i, 0, 0, preLayer->getOutputs()->channels);

		len += area;
	}

	batch = Config::instance()->getBatchSize();

	outputs  = new cuMatrix<float>(batch, len, 1);
	curDelta = new cuMatrix<float>(batch, len, 1);

	inputs.toGpu();
	preDelta.toGpu();
	inputsSkip->toGpu();
	inputsCols->toGpu();
	inputsChannels->toGpu();

	Layers::instance()->set(m_name, this);
}

/* 
 * dim3 block = dim3(batch, skip->getLen());
 * dim3 thread= dim3(min(outputs->getLen() / batch, 1024));
*/

__global__ void g_CombineLayer_feedforward(
	float** inputs,
	float* outputs,
	int* skip,
	int* cols,
	int* channels,
	int batch,
	int outputCols)
{
	int batchId     = blockIdx.x;
	int skipId      = blockIdx.y;
	int curcols     = cols[skipId];/*current input's one image feature size*/
	int curChannels = channels[skipId];

	float* output  = outputs + batchId * outputCols + skip[skipId];
	float* input   = inputs[skipId];

	int cols_channels = curChannels * curcols;
	for(int i = 0; i < cols_channels; i += blockDim.x){
		int idx = i + threadIdx.x;
		if(idx < cols_channels){
			int channel = idx / curcols;
			int col     = idx % curcols;
			int area    = batch * curcols;
			output[idx] = input[channel * area + batchId * curcols + col];
		}
	}
}

/*
 * dim3 block = dim3(batch, inputsSkip->getLen());
 * dim3 thread= dim3(min(preDelta[0]->getLen() / batch, 1024));
*/

__global__ void g_CombineLayer_backpropagation(
	float** preDeltas,
	float* curDeltas,
	int* skip,
	int* cols,
	int* channels,
	int batch,
	int curDeltaCols){
		int batchId      = blockIdx.x;
		int skipId       = blockIdx.y;

		int precols      = cols[skipId];/*current input's one image feature size*/
		int preChannels  = channels[skipId];

		float* curDelta = curDeltas + batchId * curDeltaCols + skip[skipId];
		float* preDelta = preDeltas[skipId];

		int cols_channels= precols * preChannels;

		for(int i = 0; i < cols_channels; i += blockDim.x){
			int idx = i + threadIdx.x;
			if(idx < cols_channels){
				int channel = idx / precols;
				int col     = idx % precols;
				int area    = batch * precols;
				preDelta[channel * area + batchId * precols + col] = curDelta[idx];
			}
		}
}
