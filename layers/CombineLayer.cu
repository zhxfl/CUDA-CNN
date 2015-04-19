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
	double** preDeltas,
	double* curDeltas,
	int* skip,
	int* cols,
	int curDeltaCols);

/* 
 * dim3 block = dim3(batch, skip->getLen());
 * dim3 thread= dim3(min(outputs->getLen() / batch, 1024));
*/

__global__ void g_CombineLayer_feedforward(
	double** inputs,
	double* outputs,
	int* skip,
	int* len,
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
		outputs->cols);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("CombineLayer feedforward");

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
		curDelta->cols);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("CombineLayer backpropagation");
}

CombineLayer::CombineLayer(std::string name)
{	
	cost = NULL;
	m_name = name;
	ConfigCombineLayer* config = (ConfigCombineLayer*)Config::instance()->getLayerByName(m_name);
	
	Assert(config->m_input == std::string("NULL"));
	
	/*multi-inputs*/
	/*suppose the input certainly not the BranLayers's sub-output*/

	inputsSkip = new cuMatrix<int>(config->m_inputs.size(), 1, 1);
	inputsCols = new cuMatrix<int>(config->m_inputs.size(), 1, 1);
	int len = 0;
	for(int i = 0; i < config->m_inputs.size(); i++){
		ConvLayerBase * preLayer = (ConvLayerBase*)Layers::instance()->get(config->m_inputs[i]);
		inputs.push_back(preLayer->getOutputs());
		preDelta.push_back(preLayer->getCurDelta());

		inputsSkip->set(i, 0, 0, len);
		int cols = preLayer->getOutputs()->cols * preLayer->getOutputs()->channels;
		inputsCols->set(i, 0, 0, cols);

		len += cols;
	}

	//printf("CombineLayer len %d", len);


	batch = Config::instance()->getBatchSize();

	outputs  = new cuMatrix<double>(batch, len, 1);
	curDelta = new cuMatrix<double>(batch, len, 1);

	inputs.toGpu();
	preDelta.toGpu();
	inputsSkip->toGpu();
	inputsCols->toGpu();

	Layers::instance()->set(m_name, this);
}

/* 
 * dim3 block = dim3(batch, skip->getLen());
 * dim3 thread= dim3(min(outputs->getLen() / batch, 1024));
*/

__global__ void g_CombineLayer_feedforward(
	double** inputs,
	double* outputs,
	int* skip,
	int* cols,
	int outputCols)
{
	int batchId = blockIdx.x;
	int skipId  = blockIdx.y;
	int curcols = cols[skipId];/*current input's one image feature size*/

	double* output = outputs + batchId * outputCols + skip[skipId];
	double* input  = inputs[skipId] + batchId * curcols;
	
	for(int i = 0; i < curcols; i += blockDim.x){
		int idx = i + threadIdx.x;
		if(idx < curcols){
			output[idx] = input[idx];
		}
	}
}

/*
 * dim3 block = dim3(batch, inputsSkip->getLen());
 * dim3 thread= dim3(min(preDelta[0]->getLen() / batch, 1024));
*/

__global__ void g_CombineLayer_backpropagation(
	double** preDeltas,
	double* curDeltas,
	int* skip,
	int* cols,
	int curDeltaCols)
{
	int batchId = blockIdx.x;
	int skipId  = blockIdx.y;

	int precols = cols[skipId];/*current input's one image feature size*/

	double* curDelta = curDeltas + batchId * curDeltaCols + skip[skipId];
	double* preDelta = preDeltas[skipId] + batchId * precols;

	for(int i = 0; i < precols; i += blockDim.x){
		int idx = i + threadIdx.x;
		if(idx < precols){
			preDelta[idx] = curDelta[idx];
		}
	}
}
