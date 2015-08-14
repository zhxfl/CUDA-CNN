#include "ConvCFM.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
#include "../layers/BranchLayer.h"


/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(min(kernelSize * kernelSize, 512), cfm);
*/

__global__ void g_ConvCFM_wgrad_1(float*_inputs,
	float* _curDelta,
	float** wgradTmp,
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
	float lambda);

__global__ void g_ConvCFM_feedforward_shared(
	float*  inputs,
	float** ws,
	float** bs,
	float*  outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int inputAmount,
	int outputAmount,
	int inputArea,
	int outputArea,
	int cfm);

/*
 * dim3 block = dim3(batch, outputAmount, cfm);
 * dim3 thread= min(32, kernelSize * kernelSize);
 * kernelSize = 5 || kernelSIze = 3
*/

template <int KERNELSIZE, int THREADS>
__global__ void g_ConvCFM_wgrad_2(float*_inputs,
	float* _curDelta,
	float** wgradTmp,
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
	float lambda);


/*
 * dim3 block = dim3(batch, inputAmount);
 * dim3 thread= min(inputDim * inputDim, 512);
*/
__global__ void g_ConvCFM_backpropagation_shared(
	float* _curDelta,
	float**ws,
	float* _preDelta,
	int     curDim,
	int     preDim,
	int     preAmount,
	int     curAmount,
	int     kernelSize,
	int     padding,
	int     curArea,
	int     preArea,
	int     cfm);

/*
*	blocks : dim3(batch, cuKernelScan[0]),
*	threads: dim3(min(convOutputSize * convOutputSize, 512));
*/
__global__ void g_ConvCFM_feedforward(
	float*  inputs,
	float** ws,
	float** bs,
	float*  outputs,
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
	float* _curDelta,
	float**ws,
	float* _preDelta,
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
__global__ void g_ConvCFM_wgrad(float*_inputs,
	float* _curDelta,
	float** wgradTmp,
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
	float lambda);

/*
*blocks  : dim3(kernelAmount2)
*threads : dim3(256)
*shared  : sizeof(float) * 256
*/
__global__ void g_ConvCFM_Bgrad(float* delta,
	float** bgrad,
	int deltaSize,
	int kernelAmount2,
	int batch,
	int deltaArea);


void ConvCFM::calCost()
{
	cost->gpuClear();
	g_getCost_3<<<dim3(w.size()), dim3(32), sizeof(float) * 32>>>(cost->getDev(), 
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
	
	if(inputDim * inputDim >= 512 && inputDim * inputDim <= 1024){
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= dim3(min(outputDim * outputDim, 1024));

		g_ConvCFM_feedforward_shared<<<block, thread, sizeof(float) * (inputDim * inputDim)>>>(
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
		getLastCudaError("convCFM::g_ConvCFM_feedforward_shared");
	}
	else{
		int outputDim2 = outputDim * outputDim;
		int remain = min(1024 / outputDim2, outputAmount); //32
		dim3 thread= dim3(outputDim2, remain);

		int div = (outputAmount + remain - 1) / remain;//1
		dim3 block = dim3(batch, div);

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
		getLastCudaError("convCFM::g_ConvCFM_feedforward");
	}

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

	if(inputDim * inputDim >= 512 && inputDim * inputDim <= 1024){
		dim3 block = dim3(batch, inputAmount);
		dim3 thread= min(inputDim * inputDim, 1024);

		g_ConvCFM_backpropagation_shared<<<block, thread, sizeof(float)* (outputDim * outputDim)>>>(
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
		getLastCudaError("ConvCFM::g_ConvCFM_backpropagation_shared");
	}
	else{
		int inputDim2 = inputDim * inputDim;
		int remain = min(1024 / inputDim2, inputAmount);

		int div = (inputAmount + remain - 1) / remain;
		dim3 block = dim3(batch, div);
		dim3 thread= dim3(inputDim2, remain);

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
	int ok = blockIdx.x;
	int kid= blockIdx.y;
	int c  = blockIdx.z;
	int tid = threadIdx.x;
	_sum[tid] = 0;
	__syncthreads();
	int tlen = batch;
	float* wgradTmp = _WgradTmp[ok];
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
	if(kernelSize * kernelSize * cfm < 1024){
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= dim3(kernelSize * kernelSize, cfm);
		g_ConvCFM_wgrad_1<<<block, thread>>>(
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
		getLastCudaError("g_ConvCFM_wgrad_1");
	}
	else{
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
	}
	

	dim3 block  = dim3(outputAmount, kernelSize * kernelSize, cfm);
	dim3 thread = dim3(batch);

	g_ConvCFM_wgradAdd<<<block, thread, sizeof(float) * batch>>>(
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
		sizeof(float) * thread.x>>>(curDelta->getDev(),
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
		Config::instance()->getLrate(), Config::instance()->getLrate());
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
		BranchLayer* bl = static_cast<BranchLayer*>(preLayer);
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
	cfm = inputAmount;
	NON_LINEARITY = config->m_nonLinearity;

	outputs  = new cuMatrix<float>(batch, outputDim * outputDim, outputAmount);
	curDelta = new cuMatrix<float>(batch, outputDim * outputDim, outputAmount);

	for(int i = 0; i < outputAmount; i++){
		w.push_back(new cuMatrix<float>(kernelSize, kernelSize, cfm));
		b.push_back(new cuMatrix<float>(1, 1, 1));
		wgrad.push_back(new cuMatrix<float>(kernelSize, kernelSize, cfm));
		bgrad.push_back(new cuMatrix<float>(1, 1, 1));
		wgradTmp.push_back(new cuMatrix<float>(batch, kernelSize * kernelSize, cfm));
	}

	w.toGpu();
	b.toGpu();
	wgrad.toGpu();
	bgrad.toGpu();
	wgradTmp.toGpu();

	for(int i = 0; i < outputAmount; i++){
		momentum_w.push_back(new cuMatrix<float>(kernelSize, kernelSize, cfm));
		momentum_b.push_back(new cuMatrix<float>(1, 1, 1));
	}
	momentum_w.toGpu();
	momentum_b.toGpu();

	this->initRandom();
	Layers::instance()->set(m_name, this);
}

void ConvCFM::save(FILE* file)
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

void ConvCFM::clearMomentum()
{
	for(int i = 0; i < (int)momentum_b.size(); i++){
		momentum_b[i]->gpuClear();
	}
	for(int i = 0; i < (int)momentum_w.size(); i++){
		momentum_w[i]->gpuClear();
	}
}

void ConvCFM::initRandom()
{
	//srand(clock());
	float initW = Config::instance()->getLayerByName(m_name)->m_initW;

//  	for(int i = 0; i < w.size(); i++){
//  		initMatrix(w[i], initW);
//  	}

	if(Config::instance()->getLayerByName(m_name)->isGaussian()){
		for(int i = 0; i < (int)w.size(); i++){
			float epsilon = initW;
			for(int c = 0; c < w[i]->channels; c++)
			{
				float r1 = 0.5f + 4.0f * (rand()) / RAND_MAX;
				float r2 = 0.5f + 4.0f * (rand()) / RAND_MAX;
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

void ConvCFM::initFromCheckpoint(FILE* file)
{
	float val = 0;
	for(int a = 0; a < (int)w.size(); a++){
		for(int c = 0; c < w[a]->channels; c++){
			for(int i = 0; i < w[a]->rows; i++){
				for(int j = 0; j < w[a]->cols; j++){
					fscanf(file, "%f", &val);
					w[a]->set(i, j, c, val);
				}
			}
		}

		for(int c = 0; c < b[a]->channels; c++){
			for(int i = 0; i < b[a]->rows; i++){
				for(int j = 0; j < b[a]->cols; j++){
					fscanf(file, "%f", &val);
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
	float*  inputs,
	float** ws,
	float** bs,
	float*  outputs,
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
// 	if(threadIdx.x == 0 && blockIdx.x == 0)
// 	printf("%d blockIdx.y = %d blockDim.y %d + threadIdx.y %d\n", ok, blockIdx.y, blockDim.y, threadIdx.y);
	int outputSize2 = outputDim * outputDim;
	int inputSize2  = inputDim* inputDim;
	int kernelSize2 = kernelSize * kernelSize;

	float  b       = bs[ok][0];

	float* curOutput = outputs + ok * outputArea + sp * outputSize2;

	/*convolution*/
	for(int tidx = 0; tidx < outputSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < outputSize2)
		{
			int x = idx / outputDim;
			int y = idx % outputDim;
			
			float val = 0.0;

			for(int c = 0; c < cfm; c++){
				float* curInput = inputs + c * inputArea + sp * inputSize2;
				float* w = ws[ok] + c * kernelSize2;
				/*put curInput and w into shared memory*/
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
 * dim3 block = dim3(batch, outputAmpunt);
 * dim3 thread= dim3(outputDim * outputDim);
*/

__global__ void g_ConvCFM_feedforward_shared(
	float*  inputs,
	float** ws,
	float** bs,
	float*  outputs,
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
	int ok = blockIdx.y;

	extern __shared__ float curInputS[];
	
	int outputSize2 = outputDim * outputDim;
	int inputSize2  = inputDim* inputDim;
	int kernelSize2 = kernelSize * kernelSize;

	float b = bs[ok][0];

	float* curOutput = outputs + ok * outputArea + sp * outputSize2;

	/*convolution*/
	for(int tidx = 0; tidx < outputSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < outputSize2)
		{
			int x = idx / outputDim;
			int y = idx % outputDim;

			float val = 0.0;

			for(int c = 0; c < cfm; c++){
				float* curInput = inputs + c * inputArea + sp * inputSize2;
				float* w = ws[ok] + c * kernelSize2;

				/*load curInputs*/
				for(int li = 0; li < inputSize2; li += blockDim.x){
					int lix = li + threadIdx.x;
					if(lix < inputSize2){
						curInputS[lix] = curInput[lix];
					}
				}
				__syncthreads();
				/*put curInput and w into shared memory*/
				for(int i = 0; i < kernelSize; i++){
					int xx = x + i - padding;
					for(int j = 0; j < kernelSize; j++){
						int yy = y + j - padding;
						if(xx >= 0 && xx < inputDim && yy >= 0 && yy < inputDim)
							val += curInputS[xx * inputDim + yy] * w[i * kernelSize + j];
					}
				}
			}
			curOutput[idx] = val + b;
		}
	}
}


/*
 * dim3 block = dim3(batch, inputAmount);
 * dim3 thread= min(inputDim * inputDim, 512);
*/
__global__ void g_ConvCFM_backpropagation(
	float* _curDelta,
	float**ws,
	float* _preDelta,
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
	int ik = blockIdx.y * blockDim.y + threadIdx.y;

	if(ik >= preAmount)
		return;

	int curSize2    = curDim     * curDim;
	int preSize2    = preDim     * preDim;
	int kernelSize2 = kernelSize * kernelSize;

	float *preDelta = _preDelta + ik * preArea + sp * preSize2;
	for (int tidx = 0; tidx < preSize2 + blockDim.x - 1; tidx += blockDim.x) {
		int idx = tidx + threadIdx.x;
		if (idx < preSize2) {
			int i = idx / preDim;
			int j = idx % preDim;
			float val = 0.0;
			int ok = 0;
            int c = ik;
			while(ok < curAmount){
				float *curDelta = _curDelta + ok * curArea + sp * curSize2;
				float *w = ws[ok] + c * kernelSize2;
				for (int x = 0; x < kernelSize; x++) {
					int cx = i - x + padding;
					for (int y = 0; y < kernelSize; y++) {
						int cy = j - y + padding;
						if(cx >= 0 && cx < curDim && cy >= 0 && cy < curDim){
							val += curDelta[cx * curDim + cy] * w[x * kernelSize + y];
						}
					}
				}
				ok += 1;
			}
			preDelta[idx] = val;
		}
	}
}

/*
 * dim3 block = dim3(batch, inputAmount);
 * dim3 thread= min(inputDim * inputDim, 512);
*/
__global__ void g_ConvCFM_backpropagation_shared(
	float* _curDelta,
	float**ws,
	float* _preDelta,
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
	int ik = blockIdx.y;

	int curSize2    = curDim     * curDim;
	int preSize2    = preDim     * preDim;
	int kernelSize2 = kernelSize * kernelSize;

	extern __shared__ float curDeltaS[];

	float *preDelta = _preDelta + ik * preArea + sp * preSize2;
	for (int tidx = 0; tidx < preSize2; tidx += blockDim.x) {
		int idx = tidx + threadIdx.x;
		if (idx < preSize2) {
			int i = idx / preDim;
			int j = idx % preDim;
			float val = 0.0;
			int c = ik;	
            int ok = c;
		    while(ok < curAmount){
				float *curDelta = _curDelta + ok * curArea + sp * curSize2;
				float *w = ws[ok] + c * kernelSize2;
					
				/*load curDelta*/
				for(int li = 0; li < curSize2; li += blockDim.x){
					int lix = li + threadIdx.x;
					if(lix < curSize2){
						curDeltaS[lix] = curDelta[lix];
					}
                }

				__syncthreads();

				for (int x = 0; x < kernelSize; x++) {
					int cx = i - x + padding;
					for (int y = 0; y < kernelSize; y++) {
					    int cy = j - y + padding;
						if(cx >= 0 && cx < curDim && cy >= 0 && cy < curDim){
							val += curDeltaS[cx * curDim + cy] * w[x * kernelSize + y];
						}
					}
				}
				ok += 1;
			}
			preDelta[idx] = val;
		}
	}
}


/*
 * dim3 block = dim3(batch, outputAmount, cfm);
 * dim3 thread= min(kernelSize * kernelSize, 512);
*/

__global__ void g_ConvCFM_wgrad(float*_inputs,
	float* _curDelta,
	float** wgradTmp,
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
	float lambda)
{
	int ok = blockIdx.y;
	int c  = blockIdx.z;
	int ik = c;
	int b  = blockIdx.x;

	int inputSize2    = inputDim * inputDim;
	int curDeltaSize2 = curDeltaDim * curDeltaDim;
	int kernelSize2   = kernelSize * kernelSize;

	float* wgrad = wgradTmp[ok] + c * wgradTmpArea + b * kernelSize2;

	float* input    = _inputs + inputArea * ik + b * inputSize2;
	float* curDelta = _curDelta + ok * curDeltaAea + b * curDeltaSize2;

	for(int tidx = 0; tidx < kernelSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < kernelSize2)
		{
			int i = idx / kernelSize;
			int j = idx % kernelSize;
			float val = 0.0;

			/**/
			for(int x = 0; x < curDeltaDim; x++){
				int cx = i + x - padding;
				for(int y = 0; y < curDeltaDim; y++)
				{
					int cy = j + y - padding;
					/*loader input and curDelta to shared memory*/
					if(cx >= 0 &&  cy >= 0 && cx < inputDim && cy < inputDim)
						val += input[cx * inputDim + cy] * curDelta[x * curDeltaDim + y];
				}
			}
			wgrad[idx] = val;
		}
	}
}


/*
 * dim3 block = dim3(batch, outputAmount, cfm);
 * dim3 thread= min(32, kernelSize * kernelSize);
 * kernelSize = 5 || kernelSIze = 3
*/
template <int KERNELSIZE, int THREADS>
__global__ void g_ConvCFM_wgrad_2(float*_inputs,
	float* _curDelta,
	float** wgradTmp,
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
	float lambda)
{
	extern __shared__ float _sum[KERNELSIZE * KERNELSIZE][THREADS];
	float* curSum = _sum[threadIdx.y];

	int ok = blockIdx.y;
	int c  = blockIdx.z;
	int ik = c;
	int b  = blockIdx.x;

	int inputSize2    = inputDim * inputDim;
	int curDeltaSize2 = curDeltaDim * curDeltaDim;
	int kernelSize2   = kernelSize * kernelSize;

	float* wgrad = wgradTmp[ok] + c * wgradTmpArea + b * kernelSize2;

	float* input    = _inputs + inputArea * ik + b * inputSize2;
	float* curDelta = _curDelta + ok * curDeltaAea + b * curDeltaSize2;

	for(int tidx = 0; tidx < kernelSize2; tidx += blockDim.y)
	{
		int idx = tidx + threadIdx.y;
		if(idx < kernelSize2)
		{
			int i = idx / kernelSize;
			int j = idx % kernelSize;
			float val = 0.0;
			curSum[threadIdx.x] = 0;
			
			for(int tidy = 0; tidy < curDeltaSize2; tidy += blockDim.x){
				int idy = tidy + threadIdx.x;
				if(idy < curDeltaSize2){
					int x  = idy / curDeltaDim;
					int y  = idy % curDeltaDim;
					int cx = i + x - padding;
					int cy = j + y - padding;
					if(cx >= 0 &&  cy >= 0 && cx < inputDim && cy < inputDim)
						val += input[cx * inputDim + cy] * curDelta[idy];
				}
			}
			curSum[threadIdx.x] = val;

			__syncthreads();
			int len = blockDim.x;
			while(len != 1)
			{
				__syncthreads();
				int skip = (len + 1) >> 1;
				if(threadIdx.x < (len >> 1))
				{
					curSum[threadIdx.x] += curSum[threadIdx.x + skip];
				}
				len = (len + 1) >> 1;
			}
			__syncthreads();
			if(threadIdx.x == 0)
			{
				wgrad[idx] = curSum[0];
			}
		}
	}
}



/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(min(kernelSize * kernelSize, 512), cfm);
*/

__global__ void g_ConvCFM_wgrad_1(float*_inputs,
	float* _curDelta,
	float** wgradTmp,
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
	float lambda)
{
	int ok = blockIdx.y;
	int c  = threadIdx.y;
	int ik = c;
	int b  = blockIdx.x;

	int inputSize2    = inputDim * inputDim;
	int curDeltaSize2 = curDeltaDim * curDeltaDim;
	int kernelSize2   = kernelSize * kernelSize;

	float* wgrad = wgradTmp[ok] + c * wgradTmpArea + b * kernelSize2;

	float* input    = _inputs + inputArea * ik + b * inputSize2;
	float* curDelta = _curDelta + ok * curDeltaAea + b * curDeltaSize2;

	for(int tidx = 0; tidx < kernelSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < kernelSize2)
		{
			int i = idx / kernelSize;
			int j = idx % kernelSize;
			float val = 0.0;

			/**/
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
 * shared  : sizeof(float) * 256
*/
__global__ void g_ConvCFM_Bgrad(float* delta,
	float** bgrad,
	int deltaSize,
	int kernelAmount2,
	int batch,
	int deltaArea)
{
	extern __shared__ float _sum[];
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
