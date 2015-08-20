#include "LocalConnect.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
#include "../layers/BranchLayer.h"



/*
dim3 block = dim3(batch, outputAmount);
dim3 thread= min(outputDim * outputDim, 512);
*/
__global__ void g_LocalConnect_backpropagation_kernelSize1(
	float* _curDelta,
	float**_w,
	float* _nextDelta,
	int     dim,
	int     area,
	int localKernelSize);

/*
 * block = dim3(outputAmount, kernelSize * kernelSize);
 * thread= dim3(batch);
*/
__global__ void g_LocalConnect_wgrad_Add(
	float** _WgradTmp,
	float** Wgrad,
	float** w,
	int kernelSize,
	int batch,
	float lambda,
	int wgradTmpArea,
	int wgradArea,
	int wArea);

/*
dim3 block = dim3(batch, outputAmount);
dim3 thread= min(16, min(outputDim * outputDim, 64));
*/
__global__ void g_LocalConnect_wgrad_kernelSize1(
	float* _inputs,
	float* _curDelta,
	float** _wgradTmp,
	/*float** _w,*/
	int dim,
	int area,
	int batch,
	float lambda);
/*
 *dim3 block = dim3(batch, amount);
 *dim3 thread= dim3(16, min(outputDim * outputDim, 64));
*/

__global__ void g_LocalConnect_feedforward_1(
	float** arrayS,
	float** arrayW,
	float** arrayB,
	float* _output,
	int inputSize,
	int kernelSize,
	int outputDim,
	int outputArea,
	int batch,
	int k1Amount,
	int localKernelSize);

template <int OUTPUTDIM2, int THREADS>
__global__ void g_LocalConnect_feedforward_s_2(
	float*  inputs,
	float** arrayW,
	float** arrayB,
	float* _output,
	int inputSize,
	int kernelSize,
	int outputSize,
	int inputArea,
	int outputArea,
	int batch,
	int k1Amount,
	int localKernelSize);

/*
 * function: get convolution layer and pooling output
 * dim3 block = dim3(batch, amount);
 * dim3 thread= dim3(min(outputDim * outputDim, 512));
 * const kernelsize = 1
*/

__global__ void g_LocalConnect_feedforward_kernelSize1_2(
	float*  inputs,
	float** arrayW,
	float** arrayB,
	float* _output,
	int dim,
	int area,
	int batch,
	int k1Amount,
	int localKernelSize);

/*
dim3 block = dim3(batch, outputAmount);
dim3 thread= min(outputDim * outputDim, 512);
*/
__global__ void g_LocalConnect_backpropagation(
	float* _convDelta,
	float**_w,
	float* _poolDelta,
	int     _convOutputSize,
	int     _poolOutputSize,
	int     _kernelAmount1,
	int     _kernelAmount2,
	int     _kernelSize,
	int     _convDeltaArea,
	int     _poolDeltaArea,
	int localKernelSize);
/*
 *function: get convolution layer and pooling output
 *dim3 block = dim3(batch, amount);
 *dim3 thread= dim3(min(outputDim * outputDim, 256));
*/

__global__ void g_LocalConnect_feedforward_2(
	float*  inputs,
	float** arrayW,
	float** arrayB,
	float* _output,
	int inputSize,
	int kernelSize,
	int outputSize,
	int inputArea,
	int outputArea,
	int batch,
	int k1Amount,
	int localKernelSize);

/*
* blocks  : dim3(batch, cuKernelScan[cl] * localKernelSize, Config::instance()->getChannels()),
* threads : dim3(threadidx)
*/
__global__ void g_LocalConnect_wgrad(
	float* _inputs,
	float* _curDelta,
	float** _wgrad,
	int inputDim,
	int curDeltaDim,
	int kernelSize,
	int inputAmount,
	int outputAmount,
	int inputArea,
	int curDeltaAea,
	int batch,
	float lambda);

/*
* blocks  : dim3(batch, cuKernelScan[cl], Config::instance()->getChannels()),
* threads : dim3(threadidx)
*/
__global__ void g_LocalConnect_wgrad_1(float** sArray,
	float* convDelta,
	float* WgradTmp,
	int imgSize,
	int convOutputSize,
	int kernelAmount2,
	int kernelSize,
	int sArrayArea,
	int convDeltaArea,
	int wgrapTmpArea,
	int localKernelSize);

/*
 *block = dim3(localKernelSize, amount);
 *thread= dim3(batch);
 *
*/
__global__ void g_LocalConnect_Bgrad(float* delta,
	float** bgrad,
	int deltaSize,
	int kernelAmount2,
	int batch,
	int deltaArea,
	int localKernelSize);


void LocalConnect::calCost()
{
	cost->gpuClear();
	g_getCost_3<<<dim3(w.size()), dim3(32), sizeof(float) * 32>>>(cost->getDev(), 
		w.m_devPoint, 
		lambda,
		w[0]->getLen());
	cudaStreamSynchronize(0);
	getLastCudaError("LocalConnect:getCost");
}

void LocalConnect::feedforward()
{
	if((kernelSize == 3 || kernelSize == 5) && inputDim >= 4 && inputDim <= 8){
		dim3 block = dim3(batch, outputAmount);
		const int threads = 8;
		dim3 thread= dim3(threads, outputDim * outputDim);
		if(outputDim == 4){
			g_LocalConnect_feedforward_s_2<16, threads><<<block, thread>>>(inputs->getDev(), w.m_devPoint, b.m_devPoint, outputs->getDev(), inputDim,
				kernelSize, outputDim, inputs->getArea(), outputs->getArea(), batch, outputAmount, localKernelSize);
		}else if(outputDim == 5){
			g_LocalConnect_feedforward_s_2<25, threads><<<block, thread>>>(inputs->getDev(), w.m_devPoint, b.m_devPoint, outputs->getDev(), inputDim,
				kernelSize, outputDim, inputs->getArea(), outputs->getArea(), batch, outputAmount, localKernelSize);
		}else if(outputDim == 6){
			g_LocalConnect_feedforward_s_2<36, threads><<<block, thread>>>(inputs->getDev(), w.m_devPoint, b.m_devPoint, outputs->getDev(), inputDim,
				kernelSize, outputDim, inputs->getArea(), outputs->getArea(), batch, outputAmount, localKernelSize);
		}else if(outputDim == 7){
			g_LocalConnect_feedforward_s_2<49, threads><<<block, thread>>>(inputs->getDev(), w.m_devPoint, b.m_devPoint, outputs->getDev(), inputDim,
				kernelSize, outputDim, inputs->getArea(), outputs->getArea(), batch, outputAmount, localKernelSize);
		}else if(outputDim == 8){
			g_LocalConnect_feedforward_s_2<64, threads><<<block, thread>>>(inputs->getDev(), w.m_devPoint,  b.m_devPoint, outputs->getDev(), inputDim,
				kernelSize, outputDim, inputs->getArea(), outputs->getArea(), batch, outputAmount, localKernelSize);
		}

		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("LocalConnect:g_LocalConnect_feedforward_s_2");
	}
	else if(kernelSize == 1){
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= dim3(min(outputDim * outputDim, 512));

		g_LocalConnect_feedforward_kernelSize1_2<<<block, thread>>>(
			inputs->getDev(),
			w.m_devPoint, 
			b.m_devPoint,
			outputs->getDev(),
			inputDim,
			inputs->getArea(),
			batch,
			outputAmount,
			localKernelSize);
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("LocalConnect:g_LocalConnect_feedforward_kernelSize1_2");
	}
	else {
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= dim3(8, min(outputDim * outputDim, 64));
		g_LocalConnect_feedforward_2<<<block, thread,
			sizeof(float) * outputDim * outputDim>>>
			(inputs->getDev(),
			w.m_devPoint, 
			b.m_devPoint,
			outputs->getDev(),
			inputDim,
			kernelSize,
			outputDim,
			inputs->getArea(),
			outputs->getArea(),
			batch,
			outputAmount,
			localKernelSize);
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("LocalConnect:g_LocalConnect_feedforward_2");
	}

	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));
		g_nonLinearity<<<block, thread>>>(
			outputs->getDev(), 
			outputs->getLen(),
			NON_LINEARITY);
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("LocalConnect::g_nonLinearity");
	}
}

void LocalConnect::backpropagation()
{
	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));

		g_dnonLinearity<<<block, thread>>>(curDelta->getDev(),
			outputs->getDev(), curDelta->getLen(), NON_LINEARITY);

		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("LocalConnect::g_dnonLinearity");
	}
	
	if(inputs){
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= dim3(min(outputDim * outputDim, 512));

		preDelta->gpuClear();

 		if(kernelSize == 1){
 			g_LocalConnect_backpropagation_kernelSize1<<<block, thread>>>(
 				curDelta->getDev(),
 				w.m_devPoint,
 				preDelta->getDev(),
 				outputDim,
 				curDelta->getArea(),
 				localKernelSize);
 			checkCudaErrors(cudaStreamSynchronize(0));
 			getLastCudaError("LocalConnect::g_LocalConnect_backpropagation_kernelSize1");
 
 		}else{
			g_LocalConnect_backpropagation<<<block, thread>>>(
				curDelta->getDev(),
				w.m_devPoint,
				preDelta->getDev(),
				outputDim,
				inputDim,
				inputAmount,
				outputAmount,
				kernelSize,
				curDelta->getArea(),
				preDelta->getArea(),
				localKernelSize);
			checkCudaErrors(cudaStreamSynchronize(0));
			getLastCudaError("LocalConnect::g_LocalConnect_backpropagation");
		}
	}
}


void LocalConnect::getGrad()
{
	if(kernelSize == 1){
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= dim3(min(outputDim * outputDim, 512));
		g_LocalConnect_wgrad_kernelSize1<<<block, thread, sizeof(float) * batch>>>(
			inputs->getDev(),
			curDelta->getDev(),
			wgradTmp.m_devPoint,
			inputDim,
			inputs->getArea(),
			batch,
			lambda);
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("g_LocalConnect_wgrad_kernelSize1");

		block  = dim3(outputAmount, kernelSize * kernelSize);
		thread = dim3(batch);
	}
	else{
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= min(9, min(outputDim * outputDim, 64));
		g_LocalConnect_wgrad<<<block, thread, sizeof(float) * inputDim * inputDim>>>(
			inputs->getDev(),
			curDelta->getDev(),
			wgradTmp.m_devPoint,
			inputDim,
			outputDim,
			kernelSize,
			inputAmount,
			outputAmount,
			inputs->getArea(),
			curDelta->getArea(),
			batch,
			lambda);

		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("g_LocalConnect_wgrad");
	}

	dim3 block  = dim3(outputAmount * localKernelSize, kernelSize * kernelSize);
	dim3 thread = dim3(batch);
	g_LocalConnect_wgrad_Add<<<block, thread, sizeof(float) * batch>>>(
		wgradTmp.m_devPoint,
		wgrad.m_devPoint,
		w.m_devPoint,
		kernelSize,
		batch,
		lambda,
		wgradTmp[0]->getArea(),
		wgrad[0]->getArea(),
		w[0]->getArea());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_LocalConnect_wgrad_Add");

	block = dim3(localKernelSize, outputAmount);
	thread= dim3(batch);
	g_LocalConnect_Bgrad<<<block,thread,sizeof(float) * batch>>>
		(curDelta->getDev(),
		bgrad.m_devPoint,
		outputDim,
		outputAmount,
		batch,
		curDelta->getArea(),
		localKernelSize);

	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("LocalConnect::getGrad::g_LocalConnect_Bgrad");
}

void LocalConnect::updateWeight()
{
	dim3 thread = min(256, w[0]->getLen());
	dim3 block  = momentum_w.size();
	g_vecAdd<<<block, thread>>>(momentum_w.m_devPoint, wgrad.m_devPoint, w.m_devPoint,
		momentum_b.m_devPoint, bgrad.m_devPoint, b.m_devPoint,
		w[0]->getLen(), b[0]->getLen(), 
		Config::instance()->getMomentum(),
		Config::instance()->getLrate(), Config::instance()->getLrate());
}

LocalConnect::LocalConnect(std::string name)
{
	m_name = name;
	ConfigLocal* config = static_cast<ConfigLocal*>(Config::instance()->getLayerByName(m_name));
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

	inputAmount = preLayer->outputAmount;
	outputAmount = inputAmount;
	kernelSize = config->m_kernelSize;

	inputDim  = preLayer->outputDim;
	outputDim = inputDim;
	batch     = Config::instance()->getBatchSize();
	lambda    = config->m_weightDecay;
	NON_LINEARITY = config->m_nonLinearity;

	localKernelSize = outputDim * outputDim;
	outputs = new cuMatrix<float> (batch, outputDim * outputDim, outputAmount);
	curDelta = new cuMatrix<float>(batch, outputDim * outputDim, outputAmount);

	for(int i = 0; i < outputAmount * localKernelSize; i++){
		w.push_back(new cuMatrix<float>(kernelSize, kernelSize, 1));
		b.push_back(new cuMatrix<float>(1, 1, 1));
		wgrad.push_back(new cuMatrix<float>(kernelSize, kernelSize, 1));
		bgrad.push_back(new cuMatrix<float>(1, 1, 1));
		wgradTmp.push_back(new cuMatrix<float>(batch, kernelSize * kernelSize, 1));
	}

	w.toGpu();
	b.toGpu();
	wgrad.toGpu();
	bgrad.toGpu();
	wgradTmp.toGpu();

	for(int i = 0; i < outputAmount * localKernelSize; i++){
		momentum_w.push_back(new cuMatrix<float>(kernelSize, kernelSize, 1));
		momentum_b.push_back(new cuMatrix<float>(1, 1, 1));
	}
	momentum_w.toGpu();
	momentum_b.toGpu();

	this->initRandom();
	Layers::instance()->set(m_name, this);
}

void LocalConnect::save(FILE* file)
{
	for(int a = 0; a < (int)w.size(); a++){
		w[a]->toCpu();
		b[a]->toCpu();
		for(int c = 0; c < w[a]->channels; c++){
			for(int i = 0; i < w[a]->rows; i++){
				for(int j = 0; j < w[a]->cols; j++){
					fprintf(file, "%f ", w[a]->get(i, j, c));
				}
			}
		}

		for(int c = 0; c < b[a]->channels; c++){
			for(int i = 0; i < b[a]->rows; i++){
				for(int j = 0; j < b[a]->cols; j++){
					fprintf(file, "%f ", b[a]->get(i, j, c));
				}
			}
		}
	}
}

void LocalConnect::clearMomentum()
{
	for(int i = 0; i < (int)momentum_b.size(); i++){
		momentum_b[i]->gpuClear();
	}
	for(int i = 0; i < (int)momentum_w.size(); i++){
		momentum_w[i]->gpuClear();
	}
}

void LocalConnect::initRandom()
{
	//srand(clock());
	float initW = Config::instance()->getLayerByName(m_name)->m_initW;

	if(Config::instance()->getLayerByName(m_name)->isGaussian()){
		for(int i = 0; i < (int)w.size(); i++){
			float epsilon = initW;
			for(int c = 0; c < w[i]->channels; c++)
			{
				float r1 = 0.01f + 5.0f * (rand()) / RAND_MAX;
				float r2 = 0.01f + 5.0f * (rand()) / RAND_MAX;
				createGaussian(w[i]->getHost() + c * w[i]->getArea(), r1,r2,
					kernelSize, kernelSize, w[i]->channels,
					epsilon);
			}
			w[i]->toGpu();
		}
	}
	else{
		for(int i = 0; i < (int)w.size(); i++){
			for(int j = 0; j < w[i]->getLen(); j++){
				w[i]->getHost()[j] =  initW * (2.0f * rand() / RAND_MAX - 1.0f);
				//printf("%f ", w[i]->hostData[j]);
			}//printf("\n");
			w[i]->toGpu();
		}
	}
}

void LocalConnect::initFromCheckpoint(FILE* file)
{
	float val = 0;
	for(int a = 0; a < (int)w.size(); a++){
		for(int c = 0; c < w[a]->channels; c++){
			for(int i = 0; i < w[a]->rows; i++){
				for(int j = 0; j < w[a]->cols; j++){
					if(fscanf(file, "%f", &val) == EOF){
                        LOG("scanf fail", "result/log.txt");
                    }
					w[a]->set(i, j, c, val);
				}
			}
		}

		for(int c = 0; c < b[a]->channels; c++){
			for(int i = 0; i < w[a]->rows; i++){
				for(int j = 0; j < w[a]->cols; j++){
					if(fscanf(file, "%f", &val) == EOF){
                        LOG("scanf fail", "result/log.txt");
                    }
					b[a]->set(i, j, c, val);
				}
			}
		}
		w[a]->toGpu();
		b[a]->toGpu();
	}
}

/*
 *dim3 block = dim3(batch, amount);
 *dim3 thread= dim3(16, min(outputDim * outputDim, 64));
*/

__global__ void g_LocalConnect_feedforward_1(
	float** arrayS,
	float** arrayW,
	float** arrayB,
	float* _output,
	int inputSize,
	int kernelSize,
	int outputDim,
	int outputArea,
	int batch,
	int k1Amount,
	int localKernelSize)
{
	extern __shared__ float image[];
	int sp = blockIdx.x;
	int k  = blockIdx.y;

	int OutputSize2 = outputDim  * outputDim;
	int inputSize2  = inputSize  * inputSize;
	int kernelSize2 = kernelSize * kernelSize;

	float* curInput  = arrayS[sp] + k * inputSize2;
	float* curOutput = _output + outputArea * k + sp * OutputSize2;

	/*load the image to shared memory*/
	for(int i = 0; i < inputSize2; i += blockDim.x * blockDim.y){
		int id = i + threadIdx.x + threadIdx.y * blockDim.x;
		if(id < inputSize2){
			image[id] = curInput[id];
		}
	}
	__syncthreads();

	int padding = kernelSize >> 1;
	/*convolution*/
	for(int ty = 0; ty < OutputSize2; ty += blockDim.y)
	{
		int tyid = ty + threadIdx.y;
		if(tyid < OutputSize2)
		{
			int x = tyid / outputDim;
			int y = tyid % outputDim;
			float val = 0.0;
			float* w        = arrayW[k * localKernelSize + tyid];
			float  b        = arrayB[k * localKernelSize + tyid][0];

			for(int tx = 0; tx < kernelSize2; tx += blockDim.x){
				int txid = tx + threadIdx.x;
				if(txid < kernelSize2){
					int i = txid / kernelSize;
					int j = txid % kernelSize;
					int xx = x + i - padding;
					int yy = y + j - padding;
					if(xx >= 0 && xx < inputSize && yy >= 0 && yy < inputSize)
						val += image[xx * inputSize + yy] * w[i * kernelSize + j];
				}
			}
			curOutput[tyid] = val + b;
		}
	}
}


/*
 * function: get convolution layer and pooling output
 * dim3 block = dim3(batch, amount);
 * dim3 thread= dim3(8, min(outputDim * outputDim, 64));
*/

__global__ void g_LocalConnect_feedforward_2(
	float*  inputs,
	float** arrayW,
	float** arrayB,
	float* _output,
	int inputSize,
	int kernelSize,
	int outputSize,
	int inputArea,
	int outputArea,
	int batch,
	int k1Amount,
	int localKernelSize)
{
	extern __shared__ float image[];
	int sp = blockIdx.x;
	int k  = blockIdx.y;

	int outputSize2 = outputSize * outputSize;
	int inputSize2  = inputSize  * inputSize;
	int kernelSize2 = kernelSize * kernelSize;

	float* curInput  = inputs  + k * inputArea  + sp * inputSize2;
	float* curOutput = _output + k * outputArea + sp * outputSize2;

	/*load the image to shared memory*/
	for(int i = 0; i < inputSize2; i += blockDim.x * blockDim.y){
		int id = i + threadIdx.x + threadIdx.y * blockDim.x;
		if(id < inputSize2){
			image[id] = curInput[id];
			curOutput[id] = 0;
		}
	}
	__syncthreads();

	int padding = kernelSize >> 1;
	/*convolution*/
	for(int ty = 0; ty < outputSize2; ty += blockDim.y)
	{
		int tyid = ty + threadIdx.y;
		if(tyid < outputSize2)
		{
			int x = tyid / outputSize;
			int y = tyid % outputSize;
			float val = 0.0;
			float* w = arrayW[k * localKernelSize + tyid];

			for(int tx = 0; tx < kernelSize2; tx += blockDim.x){
				int txid = tx + threadIdx.x;
				if(txid < kernelSize2){
					int i = txid / kernelSize;
					int j = txid % kernelSize;
					int xx = x + i - padding;
					int yy = y + j - padding;
					if(xx >= 0 && xx < inputSize && yy >= 0 && yy < inputSize)
						val += image[xx * inputSize + yy] * w[i * kernelSize + j];
				}
			}
			atomicAdd(curOutput + tyid, val);
		}
	}

	__syncthreads();

	for(int i = 0; i < outputSize2; i += blockDim.y * blockDim.x)
	{
		int id = i + threadIdx.y * blockDim.x + threadIdx.x;
		if(id < outputSize2)
		{
			float  b = arrayB[k * localKernelSize + id][0];
			curOutput[id] += b;
		}
	}
}



/*
 * function: get convolution layer and pooling output
 * dim3 block = dim3(batch, amount);
 * dim3 thread= dim3(min(outputDim * outputDim, 512));
 * const kernelsize = 1
*/

__global__ void g_LocalConnect_feedforward_kernelSize1_2(
	float*  inputs,
	float** arrayW,
	float** arrayB,
	float* _output,
	int dim,
	int area,
	int batch,
	int k1Amount,
	int localKernelSize)
{
	int sp = blockIdx.x;
	int k  = blockIdx.y;

	int outputSize2 = dim * dim;
	int inputSize2  = dim * dim;

	float* curInput  = inputs  + k * area + sp * inputSize2;
	float* curOutput = _output + k * area + sp * outputSize2;

	/*convolution*/
	for(int ty = 0; ty < outputSize2; ty += blockDim.x)
	{
		int tyid = ty + threadIdx.x;
		if(tyid < outputSize2)
		{
			int skip = k * localKernelSize + tyid;
			float val = 0.0;
			float w = arrayW[skip][0];
			float b = arrayB[skip][0];
			val = curInput[tyid] * w + b;
			curOutput[tyid] = val ;
		}
	}
}


/*
 * function: get convolution layer and pooling output
 * dim3 block = dim3(batch, amount);
 * dim3 thread= dim3(8, min(outputDim * outputDim, 64));
 2<64, 9, 8, 8, 64>
*/
template <int OUTPUTDIM2, int THREADS>
__global__ void g_LocalConnect_feedforward_s_2(
	float*  inputs,
	float** arrayW,
	float** arrayB,
	float* _output,
	int inputSize,
	int kernelSize,
	int outputSize,
	int inputArea,
	int outputArea,
	int batch,
	int k1Amount,
	int localKernelSize)
{
	__shared__ float image[OUTPUTDIM2];
	int sp = blockIdx.x;
	int k  = blockIdx.y;
	__shared__ float convSum[OUTPUTDIM2][THREADS];

	int outputSize2 = outputSize * outputSize;
	int inputSize2  = inputSize  * inputSize;
	int kernelSize2 = kernelSize * kernelSize;

	float* curInput  = inputs  + k * inputArea  + sp * inputSize2;
	float* curOutput = _output + k * outputArea + sp * outputSize2;

	/*load the image to shared memory*/
	for(int i = 0; i < inputSize2; i += blockDim.x * blockDim.y){
		int id = i + threadIdx.x + threadIdx.y * blockDim.x;
		if(id < inputSize2){
			image[id] = curInput[id];
		}
	}
	__syncthreads();

	int padding = kernelSize >> 1;
	/*convolution*/

	for(int ty = 0; ty < outputSize2; ty += blockDim.y)
	{
		int tyid = ty + threadIdx.y;
		
		if(tyid < outputSize2)
		{
			int x = tyid / outputSize;
			int y = tyid % outputSize;
			float val = 0.0;
			float* w = arrayW[k * localKernelSize + tyid];
			float* _convSum = convSum[threadIdx.y];
			float  b = arrayB[k * localKernelSize + tyid][0];
			_convSum[threadIdx.x] = 0;
			

			for(int tx = 0; tx < kernelSize2; tx += blockDim.x){
				int txid = tx + threadIdx.x;
				if(txid < kernelSize2){
					int i = txid / kernelSize;
					int j = txid % kernelSize;
					int xx = x + i - padding;
					int yy = y + j - padding;
					if(xx >= 0 && xx < inputSize && yy >= 0 && yy < inputSize)
						val += image[xx * inputSize + yy] * w[i * kernelSize + j];
				}
			}
			_convSum[threadIdx.x] = val;
			__syncthreads();
#pragma  unroll
			for(int len = THREADS; len != 1; len = (len + 1) >> 1){
				int skip = (len + 1) >> 1;
				if(threadIdx.x < (len >> 1)) _convSum[threadIdx.x] += _convSum[threadIdx.x + skip];
				__syncthreads();
			}
			if(threadIdx.x == 0)
				curOutput[tyid] = _convSum[0] + b;
		}
	}
}


/*
dim3 block = dim3(batch, outputAmount);
dim3 thread= min(outputDim * outputDim, 512);
*/
__global__ void g_LocalConnect_backpropagation_kernelSize1(
	float* _curDelta,
	float**_w,
	float* _nextDelta,
	int     dim,
	int     area,
	int localKernelSize)
{
	int s = blockIdx.x;
	int k = blockIdx.y;

	int dim2 = dim * dim;
	int skip = k * area + s * dim2;
	float* curDelta = _curDelta  + skip;
	float* nxtDelta = _nextDelta + skip;

	for (int tidx = 0; tidx < dim2; tidx += blockDim.x) {
		int idx = tidx + threadIdx.x;
		if (idx < dim2) {
			float val = 0.0;
			float w = _w[k * localKernelSize + idx][0];
			val = curDelta[idx] * w;
			nxtDelta[idx] = val;
		}
	}
}


/*
dim3 block = dim3(batch, outputAmount);
dim3 thread= min(outputDim * outputDim, 512);
*/
__global__ void g_LocalConnect_backpropagation (
	float* _convDelta,
	float**_w,
	float* _poolDelta,
	int     _convOutputSize,
	int     _poolOutputSize,
	int     _kernelAmount1,
	int     _kernelAmount2,
	int     _kernelSize,
	int     _convDeltaArea,
	int     _poolDeltaArea,
	int localKernelSize)
{
	int curSize = _convOutputSize;
	int wSize = _kernelSize;
	int nxtSize = _poolOutputSize;

	int s = blockIdx.x;
	int k = blockIdx.y;

	int curSize2 = curSize * curSize;
	int nxtSize2 = nxtSize * nxtSize;
	float* curDelta = _convDelta + k * _convDeltaArea + s * curSize2;
	float* nxtDelta = _poolDelta + k * _poolDeltaArea + s * nxtSize2;

	int half = wSize >> 1;
	for (int tidx = 0; tidx < nxtSize2; tidx += blockDim.x) {
		int idx = tidx + threadIdx.x;
		if (idx < nxtSize2) {
			int i = idx / nxtSize;
			int j = idx % nxtSize;
			float val = 0.0;
			for (int x = 0; x < wSize; x++) {
				for (int y = 0; y < wSize; y++) {
					int cx = i + (half - x);
					int cy = j + (half - y);
					int wx = x;
					int wy = y;
					if(cx >= 0 && cx < curSize && cy >= 0 && cy < curSize){
						float* w = _w[k * localKernelSize + cx * curSize + cy];
						val += curDelta[cx * curSize + cy] * w[wx * wSize + wy];
					}
				}
			}
			nxtDelta[idx] = val;
		}
	}
}

/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(min(outputDim * outputDim, 512));
*/
__global__ void g_LocalConnect_wgrad_kernelSize1(
	float* _inputs,
	float* _curDelta,
	float** _wgradTmp,
	int dim,
	int area,
	int batch,
	float lambda)
{
	int b  = blockIdx.x;
	int k  = blockIdx.y;

	int dim2 = dim * dim;

	int skip = k * area + b * dim2;
	float* input    = _inputs + skip;
	float* curDelta = _curDelta + skip;

	for(int y = 0; y < dim2; y += blockDim.x){
		int yid = y + threadIdx.x;
		if(yid < dim2){
			skip = k * dim2 + yid;
			float val = input[yid] * curDelta[yid];
			//_wgradTmp[skip][0] = val / batch + lambda * _w[skip][0];
			_wgradTmp[skip][0] = val;
		}
	}
}

/*
 *dim3 block = dim3(batch, outputAmount);
 *dim3 thread= min(9, min(outputDim * outputDim, 64));
*/
__global__ void g_LocalConnect_wgrad(
	float* _inputs,
	float* _curDelta,
	float** _wgradTmp,
	/*float** _w,*/
	int inputDim,
	int curDeltaDim,
	int kernelSize,
	int inputAmount,
	int outputAmount,
	int inputArea,
	int curDeltaAea,
	int batch,
	float lambda)
{
	int sp = blockIdx.x;
	int k  = blockIdx.y;

	extern __shared__ float image[];

	int inputSize2    = inputDim * inputDim;
	int curDeltaSize2 = curDeltaDim * curDeltaDim;
	int kernelSize2   = kernelSize * kernelSize;

	float* input = _inputs + k * inputArea + sp * inputSize2;


	/*load the image to shared memory*/
	for(int i = 0; i < inputSize2; i += blockDim.x * blockDim.y){
		int id = i + threadIdx.x + threadIdx.y * blockDim.x;
		if(id < inputSize2){
			image[id] = input[id];
		}
		
	}
	__syncthreads();


	float* curDelta = _curDelta + k * curDeltaAea + sp * curDeltaSize2;

	int half = (kernelSize >> 1);
	for(int y = 0; y < curDeltaSize2; y += blockDim.y){
		int yid = y + threadIdx.y;
		if(yid < curDeltaSize2){
			int ox = yid / curDeltaDim;
			int oy = yid % curDeltaDim;
			float* wgrad = _wgradTmp[k * curDeltaSize2 + yid] + sp * kernelSize2;
			float  delta = curDelta[yid];
			for(int x =  0; x < kernelSize2; x+= blockDim.x){
				int xid = x + threadIdx.x;
				if(xid < kernelSize2){
					int i = xid / kernelSize;
					int j = xid % kernelSize;
				
					int rox = ox + i - half;
					int roy = oy + j - half;
					if(rox >= 0 && rox < inputDim && roy >=0 && roy < inputDim){
						float val  = image[rox * inputDim + roy] * delta;
						wgrad[xid] = val;
					}else{
						wgrad[xid] = 0;
					}
				}
			}
		}
	}
}

/*
 *block = dim3(localKernelSize, amount)
 *thread= dim3(batch)
*/
__global__ void g_LocalConnect_Bgrad(float* _delta,
	float** bgrad,
	int deltaSize,
	int kernelAmount2,
	int batch,
	int deltaArea,
	int localKernelSize)
{
	extern __shared__ float _sum[];
	int local = blockIdx.x;
	int k     = blockIdx.y;
	int sp    = threadIdx.x;

	int deltaSize2 = deltaSize * deltaSize;
	float delta = _delta[k * deltaArea + sp * deltaSize2 + local];
	_sum[sp] = delta;
	__syncthreads();

	int len = batch;
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
		bgrad[k * localKernelSize + local][0] = _sum[0] / batch;
	}
}



/*
 * block = dim3(outputAmount, kernelSize * kernelSize);
 * thread= dim3(batch);
*/
__global__ void g_LocalConnect_wgrad_Add(
	float** _WgradTmp,
	float** Wgrad,
	float** w,
	int kernelSize,
	int batch,
	float lambda,
	int wgradTmpArea,
	int wgradArea,
	int wArea)
{
	extern __shared__ float _sum[];
	int ok  = blockIdx.x;
	int kid = blockIdx.y;
	int tid = threadIdx.x;
	_sum[threadIdx.x] = 0;
	__syncthreads();
	int tlen = batch;
	float* wgradTmp = _WgradTmp[ok];
	int kernelSize2 = kernelSize * kernelSize;
	for(int i = 0; i < tlen; i += blockDim.x)
	{
		int b = i + threadIdx.x;
		if(b < tlen)
		{
			_sum[threadIdx.x] += wgradTmp[b * kernelSize2 + kid];
		}
	}
	__syncthreads();
	int len = blockDim.x;
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
		Wgrad[ok][kid] = _sum[0] / batch + w[ok][kid] * lambda;
	}
}
