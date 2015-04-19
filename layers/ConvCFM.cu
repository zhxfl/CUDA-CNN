#include "ConvCFM.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
#include "../layers/BrachLayer.h"


__global__ void g_ConvCFM_wgrad__2(double* _inputs,
	double* _curDelta,
	double** _wgrad,
	double** w,
	int inputDim,
	int curDeltaDim,
	int kernelSize,
	int inputAmount,
	int outputAmount,
	int padding,
	int inputArea,
	int curDeltaArea,
	int batch,
	double lambda);

/*
*	blocks : dim3(batch, cuKernelScan[0]),
*	threads: dim3(min(convOutputSize * convOutputSize, 512));
*/
__global__ void g_ConvCFM_feedforward(
	double*  inputs,
	double** ws,
	double** bs,
	double*  outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int inputAmount,
	int outputAmount,
	int inputAear,
	int outputAear,
	int cfm);

/*
* blocks : dim3(batch, numOfCFM * kernelAmount2)
* threads: dim3(threadidx)
*/
__global__ void g_ConvCFM_backpropagation(
	double* _curDelta,
	double**ws,
	double* _preDelta,
	int     curDim,
	int     preDim,
	int     curAmount,
	int     preAmount,
	int     kernelSize,
	int     padding,
	int     curArea,
	int     preArea,
	int     cfm);

/*
* blocks  : dim3(batch, cuKernelScan[cl]),
* threads : dim3(threadidx)
*/
__global__ void g_ConvCFM_wgrad(double*_inputs,
	double* _curDelta,
	double** wgradTmp,
	int inputDim,
	int curDeltaDim,
	int kernelSize,
	int inputAmount,
	int outputAmount,
	int padding,
	int inputArea,
	int curDeltaAea,
	int wgradTmpArea,
	int batch,
	double lambda);

/*
*blocks  : dim3(kernelAmount2)
*threads : dim3(256)
*shared  : sizeof(double) * 256
*/
__global__ void g_ConvCFM_Bgrad(double* delta,
	double** bgrad,
	int deltaSize,
	int kernelAmount2,
	int batch,
	int deltaArea);


void ConvCFM::calCost()
{
	cost->gpuClear();
	g_getCost_3<<<dim3(w.size()), dim3(32), sizeof(double) * 32>>>(cost->getDev(), 
		w.m_devPoint, 
		lambda,
		w[0]->getLen());
	cudaDeviceSynchronize();
	getLastCudaError("ConvCFM:getCost");
}

void ConvCFM::feedforward()
{
	if((inputs == NULL))
	{
		printf("ConvCFM init error\n");
		exit(0);
	}
		
	int remain = min(512 / 128, outputAmount); //32
	int div = (outputAmount + remain - 1) / remain;//1
	dim3 block = dim3(batch, div);
	dim3 thread= dim3(min(outputDim * outputDim, 128), remain);

	g_ConvCFM_feedforward<<<block, thread>>>(
		inputs->getDev(),
		w.m_devPoint,
		b.m_devPoint,
		outputs->getDev(),
		inputDim,
		kernelSize,
		padding,
		outputDim,
		inputAmount,
		outputAmount,
		inputs->getArea(),
		outputs->getArea(),
		cfm);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("convCFM::g_ConvCFM_feedforward_2");
	


	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));
		g_nonLinearity<<<block, thread>>>(
			outputs->getDev(), 
			outputs->getLen(),
			NON_LINEARITY);
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("convCFM::g_nonLinearity");
	}
}

void ConvCFM::backpropagation()
{
	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));

		g_dnonLinearity<<<block, thread>>>(curDelta->getDev(),
			outputs->getDev(), curDelta->getLen(), NON_LINEARITY);

		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("ConvCFM::g_dnonLinearity");
	}
	
	if(Config::instance()->getLayerByName(m_name)->m_input == std::string("data"))
		return;
	if(inputs){
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= min(inputDim * inputDim, 512);

		preDelta->gpuClear();

		g_ConvCFM_backpropagation<<<block, thread>>>(
			curDelta->getDev(),
			w.m_devPoint,
			preDelta->getDev(),
			outputDim,
			inputDim,
			inputAmount,
			outputAmount,
			kernelSize,
			padding,
			curDelta->getArea(),
			preDelta->getArea(),
			cfm);
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("ConvCFM::g_ConvCFM_backpropagation");
	}
}

/*
 * block = dim3(outputAmount, kernelSize * kernelSize, cfm);
 * thread= dim3(batch);
*/
__global__ void g_ConvCFM_wgradAdd(
	double** _WgradTmp,
	double** Wgrad,
	double** w,
	int kernelSize,
	int batch,
	double lambda,
	int wgradTmpArea,
	int wgradArea,
	int wArea)
{
	extern __shared__ double _sum[];
	int ok = blockIdx.x;
	int kid= blockIdx.y;
	int c  = blockIdx.z;
	int tid = threadIdx.x;
	_sum[tid] = 0;
	__syncthreads();
	int tlen = batch;
	double* wgradTmp = _WgradTmp[ok];
	int kernelSize2 = kernelSize * kernelSize;
	for(int i = 0; i < tlen; i += blockDim.x)
	{
		int b = i + threadIdx.x;
		if(b < tlen)
		{
			_sum[threadIdx.x] += wgradTmp[c * wgradTmpArea + b * kernelSize2 + kid];
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
		Wgrad[ok][kid + c * wgradArea] = _sum[0] / batch + w[ok][kid + c * wArea] * lambda;
	}
}

void ConvCFM::getGrad()
{
	dim3 block = dim3(batch, outputAmount, cfm);
	dim3 thread= min(kernelSize * kernelSize, 512);

	g_ConvCFM_wgrad<<<block, thread>>>(
		inputs->getDev(),
		curDelta->getDev(),
		wgradTmp.m_devPoint,
		inputDim,
		outputDim,
		kernelSize,
		inputAmount,
		outputAmount,
		padding,
		inputs->getArea(),
		curDelta->getArea(),
		wgradTmp[0]->getArea(),
		batch,
		lambda);

	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("g_ConvCFM_wgrad");

	block  = dim3(outputAmount, kernelSize * kernelSize, cfm);
	thread = dim3(batch);

	g_ConvCFM_wgradAdd<<<block, thread, sizeof(double) * batch>>>(
		wgradTmp.m_devPoint,
		wgrad.m_devPoint,
		w.m_devPoint,
		kernelSize,
		batch,
		lambda,
		wgradTmp[0]->getArea(),
		wgrad[0]->getArea(),
		w[0]->getArea());
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("g_ConvCFM_wgradAdd");
	

	block = dim3(outputAmount);
	thread= dim3(256);
	g_ConvCFM_Bgrad<<<block,
		thread,
		sizeof(double) * thread.x>>>
		(curDelta->getDev(),
		bgrad.m_devPoint,
		outputDim,
		outputAmount,
		batch,
		curDelta->getArea());

	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ConvCFM::getGrad::g_ConvCFM_Bgrad");
}

void ConvCFM::updateWeight()
{
	dim3 block  = outputAmount;
	dim3 thread = min(256, w[0]->getLen());
	g_vecAdd<<<block, thread>>>(momentum_w.m_devPoint, wgrad.m_devPoint, w.m_devPoint,
		momentum_b.m_devPoint, bgrad.m_devPoint, b.m_devPoint,
		w[0]->getLen(), b[0]->getLen(), 
		Config::instance()->getMomentum(),
		Config::instance()->getLrate());
}

ConvCFM::ConvCFM(std::string name)
{
	m_name = name;
	ConfigConv* config = (ConfigConv*)Config::instance()->getLayerByName(m_name);
	ConvLayerBase * preLayer = (ConvLayerBase*)Layers::instance()->get(config->m_input);

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
	inputAmount = preLayer->outputAmount;
	outputAmount = config->m_amount;
	kernelSize = config->m_kernelSize;
	padding = config->m_padding;

	inputDim  = preLayer->outputDim;
	outputDim = (inputDim + 1 - kernelSize) + padding * 2;
	batch     = Config::instance()->getBatchSize();
	lambda    = config->m_weightDecay;
	cfm = config->m_cfm;
	NON_LINEARITY = config->m_nonLinearity;

	outputs  = new cuMatrix<double>(batch, outputDim * outputDim, outputAmount);
	curDelta = new cuMatrix<double>(batch, outputDim * outputDim, outputAmount);


	for(int i = 0; i < outputAmount; i++){
		w.push_back(new cuMatrix<double>(kernelSize, kernelSize, cfm));
		b.push_back(new cuMatrix<double>(1, 1, 1));
		wgrad.push_back(new cuMatrix<double>(kernelSize, kernelSize, cfm));
		bgrad.push_back(new cuMatrix<double>(1, 1, 1));
		wgradTmp.push_back(new cuMatrix<double>(batch, kernelSize * kernelSize, cfm));
	}

	w.toGpu();
	b.toGpu();
	wgrad.toGpu();
	bgrad.toGpu();
	wgradTmp.toGpu();

	for(int i = 0; i < outputAmount; i++){
		momentum_w.push_back(new cuMatrix<double>(kernelSize, kernelSize, cfm));
		momentum_b.push_back(new cuMatrix<double>(1, 1, 1));
	}
	momentum_w.toGpu();
	momentum_b.toGpu();

	this->initRandom();
	Layers::instance()->set(m_name, this);
}

void ConvCFM::save(FILE* file)
{
	for(int a = 0; a < w.size(); a++){
		
		w[a]->toCpu();
		b[a]->toCpu();

		for(int c = 0; c < w[a]->channels; c++){
			for(int i = 0; i < w[a]->rows; i++){
				for(int j = 0; j < w[a]->cols; j++){
					fprintf(file, "%lf ", w[a]->get(i, j, c));
				}
			}
		}

		for(int c = 0; c < b[a]->channels; c++){
			for(int i = 0; i < b[a]->rows; i++){
				for(int j = 0; j < b[a]->cols; j++){
					fprintf(file, "%lf ", b[a]->get(i, j, c));
				}
			}
		}
	}
}

void ConvCFM::clearMomentum()
{
	for(int i = 0; i < momentum_b.size(); i++){
		momentum_b[i]->gpuClear();
	}
	for(int i = 0; i < momentum_w.size(); i++){
		momentum_w[i]->gpuClear();
	}
}

void ConvCFM::initRandom()
{
	srand(clock());
	double initW = Config::instance()->getLayerByName(m_name)->m_initW;

//  	for(int i = 0; i < w.size(); i++){
//  		initMatrix(w[i], initW);
//  	}

	if(Config::instance()->getLayerByName(m_name)->isGaussian()){
		for(int i = 0; i < w.size(); i++){
			double epsilon = initW;
			for(int c = 0; c < w[i]->channels; c++)
			{
				double r1 = 0.5 + 4.0 * (rand()) / RAND_MAX;
				double r2 = 0.5 + 4.0 * (rand()) / RAND_MAX;
				createGaussian(w[i]->getHost() + c * w[i]->getArea(), r1,r2,
					kernelSize, kernelSize, w[i]->channels,
					epsilon);
			}
			w[i]->toGpu();
		}
	}
	else{
		for(int i = 0; i < w.size(); i++){
			for(int j = 0; j < w[i]->getLen(); j++){
				w[i]->getHost()[j] =  initW * (2.0 * rand() / RAND_MAX - 1.0);
				//printf("%lf ", w[i]->hostData[j]);
			}//printf("\n");
			w[i]->toGpu();
		}
	}

 		 	
}

void ConvCFM::initFromCheckpoint(FILE* file)
{
	double val = 0;
	for(int a = 0; a < w.size(); a++){
		for(int c = 0; c < w[a]->channels; c++){
			for(int i = 0; i < w[a]->rows; i++){
				for(int j = 0; j < w[a]->cols; j++){
					fscanf(file, "%lf", &val);
					w[a]->set(i, j, c, val);
				}
			}
		}

		for(int c = 0; c < b[a]->channels; c++){
			for(int i = 0; i < b[a]->rows; i++){
				for(int j = 0; j < b[a]->cols; j++){
					fscanf(file, "%lf", &val);
					b[a]->set(i, j, c, val);
				}
			}
		}
		w[a]->toGpu();
		b[a]->toGpu();
	}
}

/*
 * dim3 block = dim3(batch, div);
 * dim3 thread= dim3(min(outputDim * outputDim, 512, remain));
*/

__global__ void g_ConvCFM_feedforward(
	double*  inputs,
	double** ws,
	double** bs,
	double*  outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int inputAmount,
	int outputAmount,
	int inputArea,
	int outputArea,
	int cfm)
{
	int sp = blockIdx.x;
	int ok = blockIdx.y * blockDim.y + threadIdx.y;
	if(ok >= outputAmount)return;

	int outputSize2 = outputDim * outputDim;
	int inputSize2  = inputDim* inputDim;
	int kernelSize2 = kernelSize * kernelSize;

	double  b       = bs[ok][0];

	double* curOutput = outputs + ok * outputArea + sp * outputSize2;

	/*convolution*/
	for(int tidx = 0; tidx < outputSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < outputSize2)
		{
			int x = idx / outputDim;
			int y = idx % outputDim;
			double val = 0.0;
			for(int c = 0; c < cfm; c++){
				double* curInput = inputs + (ok + c) % inputAmount * inputArea + sp * inputSize2;
				double* w        = ws[ok] + c * kernelSize2;
				for(int i = 0; i < kernelSize; i++){
					int xx = x + i - padding;
					for(int j = 0; j < kernelSize; j++){
						int yy = y + j - padding;
						if(xx >= 0 && xx < inputDim && yy >= 0 && yy < inputDim)
							val += curInput[xx * inputDim + yy] * w[i * kernelSize + j];
					}
				}
			}
			curOutput[idx] = val + b;
		}
	}
}


/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= min(outputDim * outputDim, 512);
*/
__global__ void g_ConvCFM_backpropagation(
	double* _curDelta,
	double**ws,
	double* _preDelta,
	int     curDim,
	int     preDim,
	int     preAmount,
	int     curAmount,
	int     kernelSize,
	int     padding,
	int     curArea,
	int     preArea,
	int     cfm)
{
	int sp = blockIdx.x;
	int ok = blockIdx.y;

	int curSize2    = curDim     * curDim;
	int preSize2    = preDim     * preDim;
	int kernelSize2 = kernelSize * kernelSize;
	double *curDelta = _curDelta + ok * curArea + sp * curSize2;

	int half = kernelSize >> 1;
	for (int tidx = 0; tidx < preSize2; tidx += blockDim.x) {
		int idx = tidx + threadIdx.x;
		if (idx < preSize2) {
			int i = idx / preDim;
			int j = idx % preDim;
			for(int c = 0; c < cfm; c++){
				int ik = (ok + c) % preAmount;
				double *preDelta = _preDelta + ik * preArea + sp * preSize2;
				double *w = ws[ok] + c * kernelSize2;

				double val = 0.0;
				for (int x = 0; x < kernelSize; x++) {
					for (int y = 0; y < kernelSize; y++) {
						int cx = i + x - half;
						int cy = j + y - half;
						int wx = kernelSize - x - 1;
						int wy = kernelSize - y - 1;
						cx -= (half - padding);
						cy -= (half - padding);
						if(cx >= 0 && cx < curDim && cy >= 0 && cy < curDim){
							val += curDelta[cx * curDim + cy] * w[wx * kernelSize + wy];
						}
					}
				}
				atomicAdd(preDelta + idx, val);
			}
		}
	}
}


/*
 * dim3 block = dim3(batch, outputAmount, cfm);
 * dim3 thread= min(kernelSize * kernelSize, 512);
*/


__global__ void g_ConvCFM_wgrad(double*_inputs,
	double* _curDelta,
	double** wgradTmp,
	int inputDim,
	int curDeltaDim,
	int kernelSize,
	int inputAmount,
	int outputAmount,
	int padding,
	int inputArea,
	int curDeltaAea,
	int wgradTmpArea,
	int batch,
	double lambda)
{
	int ok = blockIdx.y;
	int c  = blockIdx.z;
	int ik = (c + ok) % inputAmount;
	int b  = blockIdx.x;

	int inputSize2    = inputDim * inputDim;
	int curDeltaSize2 = curDeltaDim * curDeltaDim;
	int kernelSize2   = kernelSize * kernelSize;

	double* wgrad = wgradTmp[ok] + c * wgradTmpArea + b * kernelSize2;

	for(int tidx = 0; tidx < kernelSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < kernelSize2)
		{
			int i = idx / kernelSize;
			int j = idx % kernelSize;
			double val = 0.0;

			double* input    = _inputs + inputArea * ik + b * inputSize2;
			double* curDelta = _curDelta + ok * curDeltaAea + b * curDeltaSize2;

			for(int x = 0; x < curDeltaDim; x++){
				int cx = i + x - padding;
				for(int y = 0; y < curDeltaDim; y++)
				{
					int cy = j + y - padding;
					if(cx >= 0 &&  cy >= 0 && cx < inputDim && cy < inputDim)
						val += input[cx * inputDim + cy] * curDelta[x * curDeltaDim + y];
				}
			}

			wgrad[idx] = val;
		}
	}
}

/*
 * blocks  : dim3(kernelAmount2)
 * threads : dim3(256)
 * shared  : sizeof(double) * 256
*/
__global__ void g_ConvCFM_Bgrad(double* delta,
	double** bgrad,
	int deltaSize,
	int kernelAmount2,
	int batch,
	int deltaArea)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	_sum[threadIdx.x] = 0.0;
	__syncthreads();
	int deltaSize2 = deltaSize * deltaSize;
	int tlen = deltaSize2 * batch;
	for(int i = 0; i < tlen; i += blockDim.x)
	{
		int idx = i + threadIdx.x;
		if(idx < tlen)
		{
			int s  = idx / (deltaSize2);//s
			int t2 = idx % (deltaSize2);//x,y
			int id =
				deltaArea * k2 + s * deltaSize2 + t2;
			_sum[threadIdx.x] += delta[id];
		}
	}
	__syncthreads();
	int len = blockDim.x;
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
		bgrad[k2][0] = _sum[0] / batch;
	}
}
