#include "Pooling.h"
#include <vector>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
#include "../common/Config.h"
#include "../common/cuBase.h"

/*
* function: unPooling
*/
__global__ void g_backpropagation(
	int* pointX,
	int* pointY,
	double* _pool,
	double* _conv,
	int poolSize,
	int convSize, 
	int poolDeltalen);

__global__ void g_feedforward(
	double* conv,
	double* pool,
	int* pointX,
	int* pointY,
	int convSize,
	int poolSize,
	int poolingSkip,
	int poolingSize,
	int convArea,
	int poolArea,
	int batch,
	int kAmount);

void Pooling::feedforward()
{
	dim3 block = dim3(batch, outputAmount);
	dim3 thread= dim3(512);
	
	g_feedforward<<<block, thread>>>(
		inputs->getDev(),
		outputs->getDev(),
		pointX->getDev(),
		pointY->getDev(),
		inputDim,
		outputDim,
		skip,
		size,
		inputs->getArea(),
		outputs->getArea(),
		batch,
		outputAmount);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pooling feedforward");

	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));
		g_nonLinearity<<<block, thread>>>(
			outputs->getDev(), 
			outputs->getLen(),
			NON_LINEARITY);
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("convNCFM::g_nonLinearity");
	}

}

void Pooling::backpropagation()
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
	dim3 block = dim3(std::min(512, (curDeltalen + 511) / 512));
	dim3 thread= dim3(512);

	g_backpropagation<<<block, thread>>>(pointX->getDev(),
		pointY->getDev(),
		curDelta->getDev(),
		preDelta->getDev(),
		outputDim,
		inputDim,
		curDeltalen);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pooling backpropagation");
}


void Pooling::getCost(cuMatrix<double>*cost, int* y)
{

}

Pooling::Pooling(std::string name)
{	
	m_name = name;
	ConfigPooling* config = (ConfigPooling*)Config::instance()->getLayerByName(m_name);
	ConvLayerBase * preLayer = (ConvLayerBase*)Layers::instance()->get(config->m_input);
	size = config->m_size;
	skip = config->m_skip;

	inputs = preLayer->getOutputs();
	inputDim = preLayer->outputDim;
	outputDim = (inputDim + skip - 1) / skip;
	amount = preLayer->outputAmount;
	inputAmount = amount;
	outputAmount = amount;
	NON_LINEARITY = config->m_nonLinearity;
	
	batch= Config::instance()->getBatchSize();
	
	outputs  = new cuMatrix<double>(batch, outputDim * outputDim, outputAmount);
	pointX   = new cuMatrix<int>   (batch, outputDim * outputDim, outputAmount);
	pointY   = new cuMatrix<int>   (batch, outputDim * outputDim, outputAmount);
	curDelta = new cuMatrix<double>(batch, outputDim * outputDim, outputAmount);
	preDelta = preLayer->getCurDelta();

	Layers::instance()->set(m_name, this);
}

/*
*blocks : dim3(batch, cuKernelScan[0]),
*threads: dim3(min(convOutputSize * convOutputSize, 512));
*/

__global__ void g_feedforward(
	double* conv,
	double* pool,
	int* pointX,
	int* pointY,
	int convSize,
	int poolSize,
	int poolingSkip,
	int poolingSize,
	int convArea,
	int poolArea,
	int batch,
	int kAmount)
{
	int sp = blockIdx.x;
	int k  = blockIdx.y;

	int convSize2  = convSize * convSize;
	int poolSize2  = poolSize * poolSize;

	int convSkip  = convArea * k + sp * convSize2;
	int poolSkip  = poolArea * k + sp * poolSize2;

	double* curConv  = conv   + convSkip;
	double* curPool  = pool   + poolSkip;
	int* px          = pointX + poolSkip;
	int* py          = pointY + poolSkip;

	/*pooling*/
	for(int tidx = 0; tidx < poolSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < poolSize2)
		{
			int x = idx / poolSize;
			int y = idx % poolSize;

			int curX = x * poolingSkip;
			int curY = y * poolingSkip;

			cuAssert(curX < convSize && curY < convSize);

			double _max = curConv[curX * convSize + curY];
			int lenx = min(convSize, curX + poolingSize);
			int leny = min(convSize, curY + poolingSize);

			for(int i = curX; i < lenx; i++)
			{
				for(int j = curY; j < leny; j++)
				{
					double val = curConv[i * convSize + j];
					if(_max < val){
						_max  = val;
						curX = i;
						curY = j;
					}
				}
			}
			px     [idx] = curX;
			py     [idx] = curY;
			curPool[idx] = _max;
		}
	}
}

/*
* function: unPooling
*/
__global__ void g_backpropagation(int* pointX, int* pointY,
	double* _pool, double* _conv,
	int poolSize, int convSize, int poolDeltalen)
{
	int poolSize2 = poolSize * poolSize;
	int convSize2 = convSize * convSize;
	for(int i = 0; i < poolDeltalen; i += gridDim.x * blockDim.x)
	{
		int id = i + blockDim.x * blockIdx.x + threadIdx.x;
		if(id < poolDeltalen)
		{
			int convId = id / poolSize2;
			int idx    = id % poolSize2;
			int poolSkip = poolSize2 * convId;
			int*       x = pointX  + poolSkip;
			int*       y = pointY  + poolSkip;
			double* pool = _pool   + poolSkip;
			double* conv = _conv   + convSize2 * convId;
			int    curX = x   [idx];
			int    curY = y   [idx];
			double curP = pool[idx];
			cuAssert(curX < convSize && curY < convSize);
			atomicAdd(conv + curX * convSize + curY, curP);
		}
	}
}
