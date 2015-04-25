#include "BranchLayer.h"
#include <vector>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
#include "../common/Config.h"
#include "../common/cuBase.h"

/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(min(outputs[0]->getLen() / batch, 1024));
*/

__global__ void g_BranchLayer_backpropagation(
	double** curDelta,
	double* preDelta,
	int curDeltaSize,
	int len);

/* 
 *dim3 block = dim3(outputs.size(), batch);
 *dim3 thread= dim3(min(outputs[0]->getLen() / batch, 1024));
*/
__global__ void g_BranchLayer_feedforward(
	double* inputs,
	double** outputs,
	int len);


void BranchLayer::feedforward()
{
	/*copy the input to outputs*/
	dim3 block = dim3(outputs.size(), batch);
	dim3 thread= dim3(min(outputs[0]->getLen() / batch, 1024));
	
	g_BranchLayer_feedforward<<<block, thread>>>(
		inputs->getDev(),
		outputs.m_devPoint,
		outputs[0]->getLen());
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("BranchLayer feedforward");
}

void BranchLayer::backpropagation()
{
	if(Config::instance()->getLayerByName(m_name)->m_input == std::string("data"))
		return;
	dim3 block = dim3(batch);
	dim3 thread= dim3(min(outputs[0]->getLen() / batch, 1024));


	preDelta->gpuClear();

	g_BranchLayer_backpropagation<<<block, thread>>>(
		curDelta.m_devPoint,
		preDelta->getDev(),
		curDelta.size(),
		preDelta->getLen());
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("BranchLayer backpropagation");
}

BranchLayer::BranchLayer(std::string name)
{	
	cost = NULL;
	m_name = name;
	ConfigBranchLayer* config = (ConfigBranchLayer*)Config::instance()->getLayerByName(m_name);
	ConvLayerBase * preLayer = (ConvLayerBase*)Layers::instance()->get(config->m_input);

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

	inputDim = preLayer->outputDim;
	outputDim = inputDim;
	outputAmount = preLayer->outputAmount;
	inputAmount = outputAmount;

	batch = Config::instance()->getBatchSize();
	for(int i = 0; i < config->m_outputs.size(); i++){
		outputs.push_back (new cuMatrix<double>(batch, outputDim * outputDim, outputAmount));
		curDelta.push_back(new cuMatrix<double>(batch, outputDim * outputDim, outputAmount));
		mapId[config->m_outputs[i]] = i;
	}

	outputs.toGpu();
	curDelta.toGpu();


	Layers::instance()->set(m_name, this);
}

/* 
 *dim3 block = dim3(outputs.size(), batch);
 *dim3 thread= dim3(min(outputs[0]->getLen() / batch, 1024));
*/
__global__ void g_BranchLayer_feedforward(
	double* inputs,
	double** outputs,
	int len)
{
	int branchId = blockIdx.x;

	double* output = outputs[branchId];

	for(int i = 0; i < len; i += gridDim.y * blockDim.x){
		int idx = i + threadIdx.x + blockIdx.y * blockDim.x;
		if(idx < len){
			output[idx] = inputs[idx];
		}
	}
}

/*
 * dim3 block = dim3(batch);
 * dim3 thread= dim3(min(outputs[0]->getLen() / batch, 1024));
*/

__global__ void g_BranchLayer_backpropagation(
	double** curDelta,
	double* preDelta,
	int curDeltaSize,
	int len)
{
	for(int i = 0; i < len ; i += gridDim.x * blockDim.x){
		int idx = i + blockDim.x * blockIdx.x + threadIdx.x;
		if(idx < len){
			double val = 0.0;
			for(int c = 0; c < curDeltaSize; c++){
				val += curDelta[c][idx];
			}
			preDelta[idx] = val;
		}
	}
}
