#include "ConvCFM.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
/*
*	blocks : dim3(batch, cuKernelScan[0], Config::instance()->getChannels()),
*	threads: dim3(min(convOutputSize * convOutputSize, 512));
*/
__global__ void g_ConvCFM_feedforward_1(
	double** arrayS,
	double** arrayW,
	double** arrayB,
	double* conv,
	int inputSize,
	int kernelSize,
	int padding,
	int convSize,
	int convArea,
	int batch,
	int k1Amount);
/*
*	blocks : dim3(batch, cuKernelScan[0], Config::instance()->getChannels()),
*	threads: dim3(min(convOutputSize * convOutputSize, 512));
*/
__global__ void g_ConvCFM_feedforward_2(
	double* pool1,
	double** arrayW,
	double** arrayB,
	double* conv2,
	int pool1Size,
	int kernelSize,
	int padding,
	int conv2Size,
	int k1Amount,
	int k2Amount,
	int pool1Area,
	int conv2Area,
	int numOfCFM);

/*
* blocks : dim3(batch, numOfCFM * kernelAmount2, Config::instance()->getChannels())
* threads: dim3(threadidx)
*/
__global__ void g_ConvCFM_backpropagation(
	double* _convDelta,
	double**_w,
	double* _poolDelta,
	int     _convOutputSize,
	int     _poolOutputSize,
	int     _kernelAmount1,
	int     _kernelAmount2,
	int     _kernelSize,
	int     _padding,
	int     _convDeltaArea,
	int     _poolDeltaArea,
	int numOfCFM);

/*
* blocks  : dim3(batch, cuKernelScan[cl], Config::instance()->getChannels()),
* threads : dim3(threadidx)
*/
__global__ void g_ConvCFM_wgrad_2(double* pool,
	double* convDelta,
	double* WgradTmp,
	int poolOutputSize,
	int convOutputSize,
	int kernelAmount1,
	int kernelAmount2,
	int kernelSize,
	int padding,
	int poolArea,
	int convDeltaArea,
	int wgradTmpArea,
	int numOfCMF);


/*
* blocks  : dim3(kernelAmount2, kernelSize * kernelSize, Config::instance()->getChannels()),
* threads : dim3(256)
* shared  : sizeof(double) * 256
*/
__global__ void g_ConvCFM_wgradAdd_2(
	double* WgradTmp,
	double** Wgrad,
	double** w,
	int kernelAmount1,
	int kernelAmount2,
	int kernelSize,
	int batch,
	int wgradTmpArea,
	int wgradArea,
	int wArea,
	double lambda,
	int numOfCFM
	);

/*
* blocks  : dim3(kernelAmount2, Config::instance()->getChannels())
* threads : dim3(256)
* shared  : sizeof(double) * 256
*/
__global__ void g_ConvCFM_Bgrad_2(double* delta,
	double** bgrad,
	int deltaSize,
	int kernelAmount2,
	int batch,
	int deltaArea);

/*
* blocks  : dim3(batch, cuKernelScan[cl], Config::instance()->getChannels()),
* threads : dim3(threadidx)
*/
__global__ void g_ConvCFM_wgrad_1(double** sArray,
	double* convDelta,
	double* WgradTmp,
	int imgSize,
	int convOutputSize,
	int kernelAmount2,
	int kernelSize,
	int padding,
	int sArrayArea,
	int convDeltaArea,
	int wgrapTmpArea);

/*
* <<<dim3(k1, kernelSize*kernelSize, channels), dim3(256)>>>
*/
__global__ void g_ConvCFM_wgradAdd_1(double* WgradTmp, double** Wgrad,
	double** w,
	int kernelAmount2,
	int kernelSize,
	int batch,
	double lambda,
	int wgradTmpArea,
	int wgradArea,
	int wArea);

/*
*blocks  : dim3(kernelAmount2, Config::instance()->getChannels())
*threads : dim3(256)
*shared  : sizeof(double) * 256
*/
__global__ void g_ConvCFM_Bgrad_1(double* delta,
	double** bgrad,
	int deltaSize,
	int kernelAmount2,
	int batch,
	int deltaArea,
	int bgradArea);


void ConvCFM::getCost(cuMatrix<double>*cost, int* y)
{
	g_getCost_3<<<dim3(amount), dim3(32), sizeof(double) * 32>>>(cost->devData, 
		w.m_devPoint, 
		lambda,
		kernelSize, 
		kernelSize);
	cudaDeviceSynchronize();
	getLastCudaError("ConvCFM:getCost");
}

void ConvCFM::feedforward()
{
	if((inputs_1 == NULL && inputs_2 == NULL) || (inputs_1 != NULL && inputs_2 != NULL))
	{
		printf("ConvCFM init error\n");
		exit(0);
	}
	if(inputs_1){
		dim3 block = dim3(batch, amount, Config::instance()->getChannels());
		dim3 thread= dim3(min(outputDim * outputDim, 512));
		g_ConvCFM_feedforward_1<<<block, thread>>>(inputs_1->m_devPoint,
			w.m_devPoint, 
			b.m_devPoint,
			outputs->devData,
			inputDim,
			kernelSize,
			padding,
			outputDim,
			outputs->getArea(),
			batch,
			amount);
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("convCFM:g_ConvCFM_feedforward_1");
	}
	else if(inputs_2){
		dim3 block = dim3(batch, amount, Config::instance()->getChannels());
		dim3 thread= dim3(min(outputDim * outputDim, 512));
		g_ConvCFM_feedforward_2<<<block, thread>>>(inputs_2->devData,
			w.m_devPoint,
			b.m_devPoint,
			outputs->devData,
			inputDim,
			kernelSize,
			padding,
			outputDim,
			inputAmount,
			outputAmount,
			inputs_2->getArea(),
			outputs->getArea(),
			cfm);
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("convCFM::g_ConvCFM_feedforward_2");
	}
	else{
		printf("ConvCFM init error\n");
		exit(0);
	}

	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));
		g_nonLinearity<<<block, thread>>>(
			outputs->devData, 
			outputs->getLen(),
			NON_LINEARITY);
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("convCFM::g_nonLinearity");
	}
}

void ConvCFM::backpropagation()
{
	if((inputs_1 == NULL && inputs_2 == NULL) || (inputs_1 != NULL && inputs_2 != NULL))
	{
		printf("ConvCFM init error\n");
		exit(0);
	}

	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));

		g_dnonLinearity<<<block, thread>>>(curDelta->devData,
			outputs->devData, curDelta->getLen(), Config::instance()->getNonLinearity());

		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("ConvCFM::g_dnonLinearity");
	}
	
	if(inputs_2){
		dim3 block = dim3(batch, cfm * outputAmount, Config::instance()->getChannels());
		dim3 thread= min(outputDim * outputDim, 512);

		preDelta->gpuClear();

		g_ConvCFM_backpropagation<<<block, thread>>>(
			curDelta->devData,
			w.m_devPoint,
			preDelta->devData,
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


void ConvCFM::getGrad()
{
	if((inputs_1 == NULL && inputs_2 == NULL) || (inputs_1 != NULL && inputs_2 != NULL))
	{
		printf("ConvCFM init error\n");
		exit(0);
	}
	if(inputs_1){
		dim3 block = dim3(batch, outputAmount, Config::instance()->getChannels());
		dim3 thread= min(kernelSize * kernelSize, 512);
		g_ConvCFM_wgrad_1<<<block, thread>>>(
			inputs_1->m_devPoint,
			curDelta->devData,
			wgradTmp->devData,
			inputDim,
			outputDim,
			amount,
			kernelSize,
			padding,
			inputDim * inputDim,
			curDelta->getArea(),
			wgradTmp->getArea());

		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("g_ConvCFM_wgrad_1");

		block = dim3(outputAmount, kernelSize * kernelSize, Config::instance()->getChannels());
		thread= dim3(256);
		g_ConvCFM_wgradAdd_1<<<block, thread,
			sizeof(double) * 256>>>(
			wgradTmp->devData,
			wgrad.m_devPoint,
			w.m_devPoint,
			outputAmount,
			kernelSize,
			batch,
			lambda,
			wgradTmp->getArea(),
			wgrad[0]->getArea(),
			w[0]->getArea());

		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("ConvCFM::getGrad::g_wgradAdd_1");

		block = dim3(amount, Config::instance()->getChannels());
		thread= dim3(256);
		g_ConvCFM_Bgrad_1<<<block,thread,sizeof(double) * 256>>>
			(curDelta->devData,
			bgrad.m_devPoint,
			outputDim,
			outputAmount,
			batch,
			curDelta->getArea(),
			bgrad[0]->getArea());

		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("ConvCFM::getGrad::g_ConvCFM_Bgrad_1");
	}
	else if(inputs_2){
		dim3 block = dim3(batch, cfm * outputAmount, Config::instance()->getChannels());
		dim3 thread= min(kernelSize * kernelSize, 512);

		g_ConvCFM_wgrad_2<<<block, thread>>>(inputs_2->devData,
			curDelta->devData,
			wgradTmp->devData,
			inputDim,
			outputDim,
			inputAmount,
			outputAmount,
			kernelSize,
			padding,
			inputs_2->getArea(),
			curDelta->getArea(),
			wgradTmp->getArea(),
			cfm
			);
		cudaDeviceSynchronize();
		getLastCudaError("g_ConvCFM_wgrad_2");

		block = dim3(amount, kernelSize * kernelSize, Config::instance()->getChannels());
		thread= dim3(256);
		g_ConvCFM_wgradAdd_2<<<block, thread, sizeof(double) * 256>>>(wgradTmp->devData,
			wgrad.m_devPoint,
			w.m_devPoint,
			inputAmount,
			outputAmount,
			kernelSize,
			batch,
			wgradTmp->getArea(),
			wgrad[0]->getArea(),
			w[0]->getArea(),
			lambda,cfm);
		cudaDeviceSynchronize();
		getLastCudaError("g_ConvCFM_wgradAdd_2");


		block = dim3(amount, Config::instance()->getChannels());
		thread= dim3(256);
		g_ConvCFM_Bgrad_2<<<block, thread, sizeof(double) * 256>>>(curDelta->devData,
			bgrad.m_devPoint,
			outputDim,
			amount,
			batch,
			curDelta->getArea());
		cudaDeviceSynchronize();
		getLastCudaError("g_ConvCFM_Bgrad_2");
	}
	else 
	{
		printf("ConvCFM init error\n");
		exit(0);
	}
}

void ConvCFM::updateWeight()
{
	dim3 thread = min(256, w[0]->getLen());
	dim3 block  = amount;
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
	if(config->m_input == std::string("data"))
	{
		inputs_1 = Layers::instance()->getInputs();
		inputs_2 = NULL;
		inputAmount = 1;
		amount = config->m_amount;
		outputAmount = amount;
		kernelSize = config->m_kernelSize;
		padding = config->m_padding;

		inputDim  = Config::instance()->getImageSize();
		outputDim = (inputDim - kernelSize + 1) + padding * 2;
		batch     = Config::instance()->getBatchSize();
		lambda = config->m_weightDecay;
		cfm = 1;
		NON_LINEARITY = Config::instance()->getNonLinearity();

		outputs  = new cuMatrix<double>(batch, outputAmount * outputDim * outputDim, Config::instance()->getChannels());
		curDelta = new cuMatrix<double>(batch, outputAmount * outputDim * outputDim, Config::instance()->getChannels());
		wgradTmp = new cuMatrix<double>(batch, cfm * outputAmount * kernelSize * kernelSize, Config::instance()->getChannels());
		preDelta = NULL;

		for(int i = 0; i < amount; i++){
			w.push_back(new cuMatrix<double>(kernelSize, kernelSize, Config::instance()->getChannels()));
			b.push_back(new cuMatrix<double>(1, 1, Config::instance()->getChannels()));
			wgrad.push_back(new cuMatrix<double>(kernelSize, kernelSize, Config::instance()->getChannels()));
			bgrad.push_back(new cuMatrix<double>(1, 1, Config::instance()->getChannels()));
		}
		w.toGpu();
		b.toGpu();
		wgrad.toGpu();
		bgrad.toGpu();

		for(int i = 0; i < amount; i++){
			momentum_w.push_back(new cuMatrix<double>(kernelSize, kernelSize, Config::instance()->getChannels()));
			momentum_b.push_back(new cuMatrix<double>(1, 1, Config::instance()->getChannels()));
		}
		momentum_w.toGpu();
		momentum_b.toGpu();
	}
	else {
		ConfigConv* config = (ConfigConv*)Config::instance()->getLayerByName(m_name);
		ConvLayerBase * preLayer = (ConvLayerBase*)Layers::instance()->get(config->m_input);

		inputs_1 = NULL;
		inputs_2 = preLayer->getOutputs();
		inputAmount = preLayer->outputAmount;
		amount = config->m_amount;
		outputAmount = amount;
		kernelSize = config->m_kernelSize;
		padding = config->m_kernelSize;

		inputDim  = preLayer->outputDim;
		outputDim = (inputDim + 1 - kernelSize) + padding * 2;
		batch     = Config::instance()->getBatchSize();
		lambda    = config->m_weightDecay;
		cfm = config->m_cfm;
		NON_LINEARITY = Config::instance()->getNonLinearity();
		
		outputs = new cuMatrix<double>(batch, outputAmount * outputDim * outputDim, Config::instance()->getChannels());
		curDelta = new cuMatrix<double>(batch, outputAmount * outputDim  * outputDim,  Config::instance()->getChannels());
		wgradTmp = new cuMatrix<double>(batch, cfm * outputAmount  * kernelSize * kernelSize, Config::instance()->getChannels());
		preDelta = preLayer->getCurDelta();

		for(int i = 0; i < amount; i++){
			w.push_back(new cuMatrix<double>(kernelSize, kernelSize, Config::instance()->getChannels()));
			b.push_back(new cuMatrix<double>(1, 1, Config::instance()->getChannels()));
			wgrad.push_back(new cuMatrix<double>(kernelSize, kernelSize, Config::instance()->getChannels()));
			bgrad.push_back(new cuMatrix<double>(1, 1, Config::instance()->getChannels()));
		}

		w.toGpu();
		b.toGpu();
		wgrad.toGpu();
		bgrad.toGpu();

		for(int i = 0; i < amount; i++){
			momentum_w.push_back(new cuMatrix<double>(kernelSize, kernelSize, Config::instance()->getChannels()));
			momentum_b.push_back(new cuMatrix<double>(1, 1, Config::instance()->getChannels()));
		}
		momentum_w.toGpu();
		momentum_b.toGpu();
	}
	this->initRandom();
	Layers::instance()->set(m_name, this);
}

void ConvCFM::save(FILE* file)
{
	for(int a = 0; a < amount; a++){
		
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
	for(int i = 0; i < w.size(); i++){
		double epsilon = 0.1;
		for(int c = 0; c < Config::instance()->getChannels(); c++)
		{
			double r1 = 0.5 + 4.0 * (rand()) / RAND_MAX;
			double r2 = 0.5 + 4.0 * (rand()) / RAND_MAX;
			createGaussian(w[i]->hostData + c * w[i]->getArea(), r1,r2,
				kernelSize, kernelSize, 
				Config::instance()->getChannels(), 
				epsilon * 0.5 + epsilon * rand() / RAND_MAX);
		}
		w[i]->toGpu();
	}
}

void ConvCFM::initFromCheckpoint(FILE* file)
{
	double val = 0;
	for(int a = 0; a < amount; a++){
		for(int c = 0; c < w[a]->channels; c++){
			for(int i = 0; i < w[a]->rows; i++){
				for(int j = 0; j < w[a]->cols; j++){
					fscanf(file, "%lf", &val);
					w[a]->set(i, j, c, val);
				}
			}
		}

		for(int c = 0; c < b[a]->channels; c++){
			fscanf(file, "%lf", &val);
			b[a]->set(0, 0, c, val);
		}
		w[a]->toGpu();
		b[a]->toGpu();
	}
}

__global__ void g_ConvCFM_feedforward_1(
	double** arrayS,
	double** arrayW,
	double** arrayB,
	double* conv,
	int inputSize,
	int kernelSize,
	int padding,
	int convSize,
	int convArea,
	int batch,
	int k1Amount)
{
	int sp = blockIdx.x;
	int k  = blockIdx.y;
	int c  = blockIdx.z;

	int convSize2  = convSize * convSize;
	int inputSize2 = inputSize* inputSize;
	int kernelSize2= kernelSize * kernelSize;

	int convSkip  = convArea * c + (sp * k1Amount + k) * convSize2;

	double* curInput = arrayS[sp] + c * inputSize2;
	double* w        = arrayW[k]  + c * kernelSize2;
	double  b        = arrayB[k][c];

	double* curConv  = conv   + convSkip;

	/*convolution*/
	for(int tidx = 0; tidx < convSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < convSize2)
		{
			int x = idx / convSize;
			int y = idx % convSize;
			double val = 0.0;
			for(int i = 0; i < kernelSize; i++)
			{
				for(int j = 0; j < kernelSize; j++)
				{
					int xx = x + i - padding;
					int yy = y + j - padding;
					if(xx >= 0 && xx < inputSize && yy >= 0 && yy < inputSize)
						val += curInput[xx * inputSize + yy] * w[i * kernelSize + j];
				}
			}
			curConv[idx] = val + b;
		}
	}
}


/*
* function: get convolution layer and pooling output
* blocks  : dim3(batch, kernelAmount2, Config::instance()->getChannels()),
* threads : dim3(threadidx)
*/


__global__ void g_ConvCFM_feedforward_2(
	double* pool1,
	double** arrayW,
	double** arrayB,
	double* conv2,
	int pool1Size,
	int kernelSize,
	int padding,
	int conv2Size,
	int k1Amount,
	int k2Amount,
	int pool1Area,
	int conv2Area,
	int numOfCFM)
{
	int sp = blockIdx.x;
	int k2 = blockIdx.y;
	int c  = blockIdx.z;

	double* w  = arrayW[k2] + kernelSize * kernelSize * c;
	double  b  = arrayB[k2][c];

	int pool1Size2 = pool1Size * pool1Size;
	int conv2Size2 = conv2Size * conv2Size;

	int skip2 = sp * k2Amount + k2;

	double* cv2 = conv2
		+ conv2Area * c
		+ skip2 * conv2Size2;

	for(int tidx = 0; tidx < conv2Size2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < conv2Size2)
		{
			int x = idx / conv2Size;
			int y = idx % conv2Size;
			double val = 0.0;
			for(int k1 = 0; k1 < numOfCFM; k1++)
			{
				int kk1 = (k1 + k2) % k1Amount;
				double* pl1 = pool1
					+ pool1Area * c
					+ (sp * k1Amount + kk1) * pool1Size2;

				for (int i = 0; i < kernelSize; i++) {
					for (int j = 0; j < kernelSize; j++) {
						int xx = x + i - padding;
						int yy = y + j - padding;
						if(xx>= 0 && xx < pool1Size && yy >= 0 && yy < pool1Size)
							val += pl1[xx * pool1Size + yy] * w[i * kernelSize + j];
					}
				}
			}
			cv2[idx] = val + b;
		}
	}
}


/*
* blocks  : dim3(batch, cuKernelScan[cl], Config::instance()->getChannels())
* threads : dim3(threadidx)
*/
__global__ void g_ConvCFM_backpropagation(
	double* _convDelta,
	double**_w,
	double* _poolDelta,
	int     _convOutputSize,
	int     _poolOutputSize,
	int     _kernelAmount1,
	int     _kernelAmount2,
	int     _kernelSize,
	int     _padding,
	int     _convDeltaArea,
	int     _poolDeltaArea,
	int numOfCFM)
{
	int curSize = _convOutputSize;
	int wSize = _kernelSize;
	int nxtSize = _poolOutputSize;
	int k2 = blockIdx.y % _kernelAmount2;
	int k1 = blockIdx.y / _kernelAmount2;

	cuAssert(k1 < numOfCFM);

	int kk1= (k1 + k2) % _kernelAmount1;

	int s = blockIdx.x;
	int c = blockIdx.z;
	int curSize2 = curSize * curSize;
	int nxtSize2 = nxtSize * nxtSize;
	double* curDelta = _convDelta + c * _convDeltaArea
		+ curSize2 * (s * _kernelAmount2 + k2);
	double* nxtDelta = _poolDelta + c * _poolDeltaArea
		+ nxtSize2 * (s * _kernelAmount1 + kk1);


	double* w = _w[k2] + c * _kernelSize * _kernelSize;

	for (int tidx = 0; tidx < nxtSize2; tidx += blockDim.x) {
		int idx = tidx + threadIdx.x;
		if (idx < nxtSize2) {
			int i = idx / nxtSize;
			int j = idx % nxtSize;
			double val = 0.0;
			for (int x = 0; x < wSize; x++) {
				for (int y = 0; y < wSize; y++) {
					int cx = i + x - (wSize >> 1);
					int cy = j + y - (wSize >> 1);
					int wx = wSize - x - 1;
					int wy = wSize - y - 1;
					cx -= ((wSize >> 1) - _padding);
					cy -= ((wSize >> 1) - _padding);
					if(cx >= 0 && cx < curSize && cy >= 0 && cy < curSize){
						val += curDelta[cx * curSize + cy] * w[wx * wSize + wy];
					}
				}
			}
			atomicAdd(nxtDelta + idx, val);
		}
	}
}


/*
* blocks  : dim3(batch, cuKernelScan[cl], Config::instance()->getChannels()),
* threads : dim3(threadidx)
*/
__global__ void g_ConvCFM_wgrad_2(double* pool,
	double* convDelta,
	double* WgradTmp,
	int poolOutputSize,
	int convOutputSize,
	int kernelAmount1,
	int kernelAmount2,
	int kernelSize,
	int padding,
	int poolArea,
	int convDeltaArea,
	int wgradTmpArea,
	int numOfCMF)
{
	int c  = blockIdx.z;
	int s  = blockIdx.x;
	int k2 = blockIdx.y % kernelAmount2;
	int k1 = blockIdx.y / kernelAmount2;

	cuAssert(k1 < numOfCMF);

	int kk1= (k1 + k2) % kernelAmount1;

	int curSize = poolOutputSize;
	int wSize   = convOutputSize;
	int nxtSize = kernelSize;
	int curSize2 = curSize * curSize;
	int wSize2   = wSize   * wSize;
	int nxtSize2 = nxtSize * nxtSize;
	double* cur   = pool
		+ c * poolArea
		+ curSize2 * (s * kernelAmount1 + kk1);
	double* w     = convDelta
		+ c * convDeltaArea
		+ wSize2 * (s * kernelAmount2 + k2);
	double* nxt   = WgradTmp
		+ c * wgradTmpArea
		+ nxtSize2 * (s * numOfCMF * kernelAmount2 + k1 * kernelAmount2 + k2);

	for(int tidx = 0; tidx < nxtSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < nxtSize2)
		{
			int i = idx / nxtSize;
			int j = idx % nxtSize;
			double val = 0.0;
			for(int x = 0; x < wSize; x++)
			{
				for(int y = 0; y < wSize; y++)
				{
					int cx = i + x - padding;
					int cy = j + y - padding;
					if(cx >= 0 && cy >= 0 && cx < curSize && cy < curSize)
						val += cur[cx * curSize + cy] * w[x * wSize + y];
				}
			}
			nxt[idx] = val;
		}
	}
}
/*
* blocks  : dim3(kernelAmount2, kernelSize * kernelSize, Config::instance()->getChannels()),
* threads : dim3(256)
* shared  : sizeof(double) * 256
*/
__global__ void g_ConvCFM_wgradAdd_2(
	double* WgradTmp,
	double** Wgrad,
	double** w,
	int kernelAmount1,
	int kernelAmount2,
	int kernelSize,
	int batch,
	int wgradTmpArea,
	int wgradArea,
	int wArea,
	double lambda,
	int numOfCFM)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	int kid= blockIdx.y;
	int c  = blockIdx.z;
	_sum[threadIdx.x] = 0;
	__syncthreads();
	int kernelSize2 = kernelSize * kernelSize;
	int  tlen = batch * numOfCFM;
	for(int i = 0; i <  tlen; i += blockDim.x)
	{
		int idx = i + threadIdx.x;
		if(idx < tlen)
		{
			int s  = idx / numOfCFM;
			int k1 = idx % numOfCFM;

			int id =
				c * wgradTmpArea
				+ kernelSize2 * (s * numOfCFM * kernelAmount2 + k1* kernelAmount2 + k2)
				+ kid;
			_sum[threadIdx.x] += WgradTmp[id];
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
		Wgrad[k2][kid + c * wgradArea] = _sum[0] / batch + w[k2][kid + c * wArea] * lambda;
	}
}
/*
* blocks  : dim3(kernelAmount2, Config::instance()->getChannels())
* threads : dim3(256)
* shared  : sizeof(double) * 256
*/
__global__ void g_ConvCFM_Bgrad_2(double* delta,
	double** bgrad,
	int deltaSize,
	int kernelAmount2,
	int batch,
	int deltaArea)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	int c  = blockIdx.y;
	_sum[threadIdx.x] = 0.0;
	__syncthreads();
	int deltaSize2 = deltaSize * deltaSize;
	int tlen = batch * deltaSize2;
	for(int i = 0; i < tlen; i += blockDim.x)
	{
		int idx = i + threadIdx.x;
		if(idx < tlen)
		{
			int s  = idx / deltaSize2;//s,kernel1
			int t2 = idx % deltaSize2;//x,y
			int id =
				c * deltaArea
				+ deltaSize2 * (s * kernelAmount2 + k2)
				+ t2;
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
		bgrad[k2][c] = _sum[0] / batch;
	}
}
/*
* blocks  : dim3(batch, cuKernelScan[cl], Config::instance()->getChannels()),
* threads : dim3(threadidx)
*/
__global__ void g_ConvCFM_wgrad_1(double** sArray,
	double* convDelta,
	double* WgradTmp,
	int imgSize,
	int convOutputSize,
	int kernelAmount2,
	int kernelSize,
	int padding,
	int sArrayArea,
	int convDeltaArea,
	int wgrapTmpArea)
{
	int curSize = imgSize;
	int wSize   = convOutputSize;
	int nxtSize = kernelSize;
	int s = blockIdx.x;
	int k2= blockIdx.y;
	int c = blockIdx.z;
	int wSize2   = wSize * wSize;
	int nxtSize2 = nxtSize * nxtSize;
	double* cur  = sArray[s] + c * sArrayArea;
	double* w     = convDelta
		+ c * convDeltaArea
		+ wSize2 * (s * kernelAmount2 + k2);
	double* nxt   = WgradTmp
		+ c * wgrapTmpArea
		+ nxtSize2 * (s * kernelAmount2 + k2);
	for(int tidx = 0; tidx < nxtSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < nxtSize2)
		{
			int i = idx / nxtSize;
			int j = idx % nxtSize;
			double val = 0.0;
			for(int x = 0; x < wSize; x++)
			{
				for(int y = 0; y < wSize; y++)
				{
					int cx = i + x - padding;
					int cy = j + y - padding;
					if(cx >= 0 &&  cy >= 0 && cx < curSize && cy < curSize)
						val += cur[cx * curSize + cy] * w[x * wSize + y];
				}
			}
			nxt[idx] = val;
		}
	}
}
/*
* <<<dim3(k1, kernelSize*kernelSize, channels), dim3(256)>>>
*/
__global__ void g_ConvCFM_wgradAdd_1(double* WgradTmp, double** Wgrad,
	double** w,
	int kernelAmount2,
	int kernelSize,
	int batch,
	double lambda,
	int wgradTmpArea,
	int wgradArea,
	int wArea)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	int kid= blockIdx.y;
	int c  = blockIdx.z;
	int tid= threadIdx.x;
	_sum[threadIdx.x] = 0;
	__syncthreads();
	int kernelSize2 = kernelSize * kernelSize;
	int tlen = batch;
	for(int i = 0; i < tlen; i += blockDim.x)
	{
		int s = i + threadIdx.x;
		if(s < tlen)
		{
			int id =
				c * wgradTmpArea
				+ kernelSize2 * (s * kernelAmount2 + k2)
				+ kid;
			_sum[threadIdx.x] += WgradTmp[id];
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
		Wgrad[k2][kid + c * wgradArea] = _sum[0] / batch + w[k2][kid + c * wArea] * lambda;
	}
}
/*
*blocks  : dim3(kernelAmount2, Config::instance()->getChannels())
*threads : dim3(256)
*shared  : sizeof(double) * 256
*/
__global__ void g_ConvCFM_Bgrad_1(double* delta,
	double** bgrad,
	int deltaSize,
	int kernelAmount2,
	int batch,
	int deltaArea,
	int bgradArea)
{
	extern __shared__ double _sum[];
	int k2 = blockIdx.x;
	int c  = blockIdx.y;
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
				deltaArea * c
				+ deltaSize2 * (s * kernelAmount2 + k2)
				+ t2;
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
		bgrad[k2][c] = _sum[0] / batch;
	}
}