#include "Pooling.h"
#include <vector>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
#include "../common/Config.h"
#include "../common/cuBase.h"
#include "../layers/BranchLayer.h"

/*
* function: unPooling
*/
__global__ void g_pooling_backpropagation_max(
	int* pointX,
	int* pointY,
	float* _pool,
	float* _conv,
	int poolSize,
	int convSize, 
	int poolDeltalen);

__global__ void g_pooling_feedforward_avr(
	float* conv,
	float* pool,
	int convSize,
	int poolSize,
	int poolingSkip,
	int poolingSize,
	int convArea,
	int poolArea,
	int batch,
	int kAmount);


/*
* function: unPooling
*/
__global__ void g_pooling_backpropagation_max_no_atomic(int* pointX, int* pointY,
	float* _pool, float* _conv,
	int poolSize, int convSize, int poolDeltalen);


/*
* function: unPooling
*/
__global__ void g_pooling_backpropagation_avr_no_atomic(float* _pool, float* _conv,
	int poolDim, int convDim, int poolingSkip, int poolingSize, int poolDeltalen);

/*
* function: unPooling
*/
__global__ void g_pooling_backpropagation_avr(float* _pool, float* _conv,
	int poolDim, int convDim, int poolingSkip, int poolingSize, int poolDeltalen);

__global__ void g_pooling_feedforward_max(
	float* conv,
	float* pool,
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

	int threadx = min(outputDim * outputDim , 512);
	if(threadx <= 16) threadx = 16;
	else if(threadx <= 256) threadx = 64;
	int remain = 1024 / threadx;
	int div = (outputAmount + remain - 1) / remain;

	dim3 block = dim3(batch, div);
	dim3 thread= dim3(threadx, remain);

	if(type == std::string("max")){
        cudaFuncSetCacheConfig(g_pooling_feedforward_max, cudaFuncCachePreferL1);
		g_pooling_feedforward_max<<<block, thread>>>(
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
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("pooling g_pooling_feedforward_max");
	}else{
		g_pooling_feedforward_avr<<<block, thread>>>(
			inputs->getDev(),
			outputs->getDev(),
			inputDim,
			outputDim,
			skip,
			size,
			inputs->getArea(),
			outputs->getArea(),
			batch,
			outputAmount);
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("pooling g_pooling_feedforward_avr");
	}
	

	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));
		g_nonLinearity<<<block, thread>>>(
			outputs->getDev(), 
			outputs->getLen(),
			NON_LINEARITY);
		checkCudaErrors(cudaStreamSynchronize(0));
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

		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("ConvNCFM::g_dnonLinearity");
	}

	if(type == std::string("max")){
		int curDeltalen = curDelta->getLen();
		int threadx = outputDim * outputDim;
		threadx = 1024 / threadx * threadx;
		dim3 block = dim3(std::min(256, (curDeltalen + threadx) / threadx));
		dim3 thread= dim3(threadx);
		
		if(size == skip){
			g_pooling_backpropagation_max_no_atomic<<<block, thread>>>(
				pointX->getDev(), pointY->getDev(), curDelta->getDev(), preDelta->getDev(),
				outputDim, inputDim, curDeltalen);
			checkCudaErrors(cudaStreamSynchronize(0));
			getLastCudaError("pooling g_backpropagation_max_no_atomic");
		}
		else{
			preDelta->gpuClear();
			g_pooling_backpropagation_max<<<block, thread>>>(
				pointX->getDev(), pointY->getDev(), curDelta->getDev(), preDelta->getDev(),
				outputDim, inputDim, curDeltalen);
			checkCudaErrors(cudaStreamSynchronize(0));
			getLastCudaError("pooling backpropagation_max");
		}
	}else{
		int curDeltalen = curDelta->getLen();
		int threadx = outputDim * outputDim;
		threadx = 1024 / threadx * threadx;
		dim3 block = dim3(std::min(256, (curDeltalen + threadx) / threadx));
		dim3 thread= dim3(threadx);
		if(size == skip){
			/*do not clear need preDelta->gpuClear();*/
			g_pooling_backpropagation_avr_no_atomic<<<block, thread>>>(
				curDelta->getDev(), preDelta->getDev(), 
				outputDim, inputDim, skip, size, curDeltalen);
			checkCudaErrors(cudaStreamSynchronize(0));
			getLastCudaError("pooling g_backpropagation_avr_no_atomic");
		}else{
			preDelta->gpuClear();
			g_pooling_backpropagation_avr<<<block, thread>>>(curDelta->getDev(), preDelta->getDev(), 
				outputDim, inputDim, skip, size, curDeltalen);
			checkCudaErrors(cudaStreamSynchronize(0));
			getLastCudaError("pooling g_pooling_backpropagation_avr");
		}
	}
	
}


Pooling::Pooling(std::string name)
{	
	cost = NULL;
	m_name = name;
	ConfigPooling* config = (ConfigPooling*)Config::instance()->getLayerByName(m_name);
	ConvLayerBase * preLayer = (ConvLayerBase*)Layers::instance()->get(config->m_input);
	type = config->m_poolingType;

	size = config->m_size;
	skip = config->m_skip;

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
	outputDim = (inputDim + skip - 1) / skip;
	inputAmount = preLayer->outputAmount;
	outputAmount = inputAmount;
	NON_LINEARITY = config->m_nonLinearity;
	
	batch= Config::instance()->getBatchSize();
	
	outputs  = new cuMatrix<float>(batch, outputDim * outputDim, outputAmount);
	curDelta = new cuMatrix<float>(batch, outputDim * outputDim, outputAmount);
	if(type == std::string("max")){
		pointX   = new cuMatrix<int>   (batch, outputDim * outputDim, outputAmount);
		pointY   = new cuMatrix<int>   (batch, outputDim * outputDim, outputAmount);
	}else{
		pointX = pointY = NULL;
	}

	Layers::instance()->set(m_name, this);
}

/*
*blocks : dim3(batch, cuKernelScan[0]),
*threads: dim3(min(convOutputSize * convOutputSize, 512));
*/

__global__ void g_pooling_feedforward_max(
	float* conv,
	float* pool,
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
	int k  = blockIdx.y * blockDim.y + threadIdx.y;
	if(k >= kAmount)return;

	int convSize2 = convSize * convSize;
	int poolSize2 = poolSize * poolSize;

	int convSkip = convArea * k + sp * convSize2;
	int poolSkip = poolArea * k + sp * poolSize2;

	float* curConv = conv + convSkip;
	float* curPool = pool + poolSkip;
	int* px = pointX + poolSkip;
	int* py = pointY + poolSkip;

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

			assert(curX < convSize && curY < convSize);

			float _max = curConv[curX * convSize + curY];
			int lenx = min(convSize, curX + poolingSize);
			int leny = min(convSize, curY + poolingSize);

			for(int i = curX; i < lenx; i++)
			{
				for(int j = curY; j < leny; j++)
				{
					float val = curConv[i * convSize + j];
					if(_max < val){
						_max  = val;
						curX = i;
						curY = j;
					}
				}
			}
			px [idx] = curX;
			py [idx] = curY;
			curPool[idx] = _max;
		}
	}
}



/*
*blocks : dim3(batch, cuKernelScan[0]),
*threads: dim3(min(convOutputSize * convOutputSize, 512));
*/

__global__ void g_pooling_feedforward_avr(
	float* conv,
	float* pool,
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
	int k  = blockIdx.y * blockDim.y + threadIdx.y;
	if(k >= kAmount)return;

	int convSize2  = convSize * convSize;
	int poolSize2  = poolSize * poolSize;

	int convSkip  = convArea * k + sp * convSize2;
	int poolSkip  = poolArea * k + sp * poolSize2;

	float* curConv  = conv   + convSkip;
	float* curPool  = pool   + poolSkip;

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

			float _sum = 0.0;
			int lenx = min(convSize, curX + poolingSize);
			int leny = min(convSize, curY + poolingSize);

			for(int i = curX; i < lenx; i++)
			{
				for(int j = curY; j < leny; j++)
				{
					float val = curConv[i * convSize + j];
					_sum += val;
				}
			}
			curPool[idx] = _sum / (poolingSize * poolingSize);
		}
	}
}

/*
* function: unPooling
*/
__global__ void g_pooling_backpropagation_max(int* pointX, int* pointY,
	float* _pool, float* _conv,
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
			float* pool = _pool   + poolSkip;
			float* conv = _conv   + convSize2 * convId;
			int    curX = x   [idx];
			int    curY = y   [idx];
			float curP = pool[idx];
			cuAssert(curX < convSize && curY < convSize);
			atomicAdd(conv + curX * convSize + curY, curP);
		}
	}
}

/*
* function: unPooling
*/
__global__ void g_pooling_backpropagation_max_no_atomic(int* pointX, int* pointY,
	float* _pool, float* _conv,
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
			float* pool = _pool   + poolSkip;
			float* conv = _conv   + convSize2 * convId;
			int    curX = x   [idx];
			int    curY = y   [idx];
			float curP = pool[idx];
			assert(curX < convSize && curY < convSize);
			conv[curX * convSize + curY] = curP;
		}
	}
}



/*
* function: unPooling
*/
__global__ void g_pooling_backpropagation_avr(float* _pool, float* _conv,
	int poolDim, int convDim, int poolingSkip, int poolingSize, int poolDeltalen)
{
	int poolSize2 = poolDim * poolDim;
	int convSize2 = convDim * convDim;
	for(int i = 0; i < poolDeltalen; i += gridDim.x * blockDim.x)
	{
		int id = i + blockDim.x * blockIdx.x + threadIdx.x;
		if(id < poolDeltalen)
		{
			int convId = id / poolSize2;
			int idx    = id % poolSize2;

			float* pool = _pool   + poolSize2 * convId;
			float* conv = _conv   + convSize2 * convId;

			int x = idx / poolDim;
			int y = idx % poolDim;

			int curX = x * poolingSkip;
			int curY = y * poolingSkip;

			int lenx = min(convDim, curX + poolingSize);
			int leny = min(convDim, curY + poolingSize);

			float val = pool[idx] / (poolingSize * poolingSize);
			for(int i = curX; i < lenx; i++)
			{
				for(int j = curY; j < leny; j++)
				{
					cuAssert(i < convDim && j < convDim);
					atomicAdd(conv + i * convDim + j, val);
				}
			}
		}
	}
}


/*
* function: unPooling
*/
__global__ void g_pooling_backpropagation_avr_no_atomic(float* _pool, float* _conv,
	int poolDim, int convDim, int poolingSkip, int poolingSize, int poolDeltalen)
{
	int poolSize2 = poolDim * poolDim;
	int convSize2 = convDim * convDim;
	for(int i = 0; i < poolDeltalen; i += gridDim.x * blockDim.x)
	{
		int id = i + blockDim.x * blockIdx.x + threadIdx.x;
		if(id < poolDeltalen)
		{
			int convId = id / poolSize2;
			int idx    = id % poolSize2;

			float* pool = _pool   + poolSize2 * convId;
			float* conv = _conv   + convSize2 * convId;

			int x = idx / poolDim;
			int y = idx % poolDim;

			int curX = x * poolingSkip;
			int curY = y * poolingSkip;

			int lenx = min(convDim, curX + poolingSize);
			int leny = min(convDim, curY + poolingSize);

			float val = pool[idx] / (poolingSize * poolingSize);
			for(int i = curX; i < lenx; i++)
			{
				for(int j = curY; j < leny; j++)
				{
					assert(i < convDim && j < convDim);
					conv[i * convDim + j] = val;
				}
			}
		}
	}
}
