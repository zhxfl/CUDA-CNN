#include "NIN.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
#include "../layers/BranchLayer.h"


/*
 * dim3 block = dim3(batch, outputAmpunt);
 * dim3 thread= dim3(outputDim * outputDim);
*/

__global__ void g_NIN_feedforward(
	float*  inputs,
	float** ws,
	float** bs,
	float*  outputs,
	int inputDim,
	int outputDim,
	int inputAmount,
	int outputAmount,
	int inputArea,
	int outputArea);

/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(THREADS, inputAmount);
*/
template <int INPUTAMOUNT, int THREADS>
__global__ void g_NIN_wgrad_1(float*_inputs,
	float* _curDelta,
	float** wgradTmp,
	int inputDim,
	int curDeltaDim,
	int inputAmount,
	int outputAmount,
	int inputArea,
	int curDeltaAea);

/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(inputAmount);
*/

__global__ void g_NIN_wgrad(float*_inputs,
	float* _curDelta,
	float** wgradTmp,
	int inputDim,
	int curDeltaDim,
	int inputAmount,
	int outputAmount,
	int inputArea,
	int curDeltaAea);

/*
 * dim3 block = dim3(batch, inputAmount);
 * dim3 thread= min(inputDim * inputDim, 512);
*/
__global__ void g_NIN_backpropagation(
	float* _curDelta,
	float**ws,
	float* _preDelta,
	int     curDim,
	int     preDim,
	int     preAmount,
	int     curAmount,
	int     curArea,
	int     preArea);

/*
 * block = dim3(outputAmount, inputAmount);
 * thread= dim3(batch);
*/
__global__ void g_NIN_wgradAdd(
	float** _WgradTmp,
	float** Wgrad,
	float** w,
	int batch,
	float lambda);

/*
*blocks  : dim3(kernelAmount2)
*threads : dim3(256)
*shared  : sizeof(float) * 256
*/
__global__ void g_NIN_Bgrad(float* delta,
	float** bgrad,
	int deltaSize,
	int kernelAmount2,
	int batch,
	int deltaArea);


void NIN::calCost()
{
	cost->gpuClear();
	g_getCost_3<<<dim3(w.size()), dim3(32), sizeof(float) * 32>>>(cost->getDev(), 
		w.m_devPoint, 
		lambda,
		w[0]->getLen());
	cudaStreamSynchronize(0);
	getLastCudaError("NIN:getCost");
}

void NIN::feedforward()
{
	if((inputs == NULL))
	{
		printf("NIN init error\n");
		exit(0);
	}

	dim3 block = dim3(batch, outputAmount);
	dim3 thread= dim3(min(outputDim * outputDim, 1024));

	g_NIN_feedforward<<<block, thread>>>(
		inputs->getDev(),
		w.m_devPoint,
		b.m_devPoint,
		outputs->getDev(),
		inputDim,
		outputDim,
		inputAmount,
		outputAmount,
		inputs->getArea(),
		outputs->getArea());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("NIN::g_NIN_feedforward");

	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));
		g_nonLinearity<<<block, thread>>>(
			outputs->getDev(), 
			outputs->getLen(),
			NON_LINEARITY);
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("NIN::g_nonLinearity");
	}
}

void NIN::backpropagation()
{
	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));

		g_dnonLinearity<<<block, thread>>>(curDelta->getDev(),
			outputs->getDev(), curDelta->getLen(), NON_LINEARITY);

		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("NIN::g_dnonLinearity");
	}
	
	if(Config::instance()->getLayerByName(m_name)->m_input == std::string("data"))
		return;

	dim3 block = dim3(batch, inputAmount);
	dim3 thread= dim3(min(inputDim * inputDim, 1024));

	g_NIN_backpropagation<<<block, thread, sizeof(float) * outputAmount>>>(
		curDelta->getDev(),
		w.m_devPoint,
		preDelta->getDev(),
		outputDim,
		inputDim,
		inputAmount,
		outputAmount,
		curDelta->getArea(),
		preDelta->getArea());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("NIN::g_NIN_backpropagation");
}

/*
 * block = dim3(outputAmount, inputAmount);
 * thread= dim3(batch);
*/
__global__ void g_NIN_wgradAdd(
	float** _WgradTmp,
	float** Wgrad,
	float** w,
	int batch,
	float lambda)
{
	extern __shared__ float _sum[];
	int ok = blockIdx.x;
	int ik = blockIdx.y;
	int tid = threadIdx.x;
	_sum[tid] = 0;
	int inputAmount = gridDim.y;
	__syncthreads();
	int tlen = batch;
	float* wgradTmp = _WgradTmp[ok];
	for(int i = 0; i < tlen; i += blockDim.x)
	{
		int b = i + threadIdx.x;
		if(b < tlen)
		{
			_sum[threadIdx.x] += wgradTmp[ik + b * inputAmount];
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
        else{
            return;
        }
		len = (len + 1) >> 1;
	}
	if(tid == 0)
	{
		Wgrad[ok][ik] = _sum[0] / batch + w[ok][ik] * lambda;
	}
}

void NIN::getGrad()
{
	/*if(outputDim >= 8 && inputAmount == 32){
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= dim3(32, inputAmount);
		g_NIN_wgrad_1<32, 32><<<block, thread>>>(
			inputs->getDev(),
			curDelta->getDev(),
			wgradTmp.m_devPoint,
			inputDim,
			outputDim,
			inputAmount,
			outputAmount,
			inputs->getArea(),
			curDelta->getArea());
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("g_NIN_wgrad_1");
	}else if(outputDim >= 8 && inputAmount == 64){
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= dim3(16, 64);
		g_NIN_wgrad_1<64, 16><<<block, thread>>>(
			inputs->getDev(),
			curDelta->getDev(),
			wgradTmp.m_devPoint,
			inputDim,
			outputDim,
			inputAmount,
			outputAmount,
			inputs->getArea(),
			curDelta->getArea());
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("g_NIN_wgrad_1");
    }else if(outputDim >=8 && inputAmount == 128){
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= dim3(8, 128);
		g_NIN_wgrad_1<128, 8><<<block, thread>>>(
			inputs->getDev(),
			curDelta->getDev(),
			wgradTmp.m_devPoint,
			inputDim,
			outputDim,
			inputAmount,
			outputAmount,
			inputs->getArea(),
			curDelta->getArea());
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("g_NIN_wgrad_1");
	}else{
    */
    {
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= dim3(inputAmount);
		g_NIN_wgrad<<<block, thread>>>(
			inputs->getDev(),
			curDelta->getDev(),
			wgradTmp.m_devPoint,
			inputDim,
			outputDim,
			inputAmount,
			outputAmount,
			inputs->getArea(),
			curDelta->getArea());

		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("g_NIN_wgrad");
    }
    //}

	dim3 block  = dim3(outputAmount, inputAmount);
	dim3 thread = dim3(batch);

	g_NIN_wgradAdd<<<block, thread, sizeof(float) * batch>>>(
		wgradTmp.m_devPoint,
		wgrad.m_devPoint,
		w.m_devPoint,
		batch,
		lambda);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("g_NIN_wgradAdd");
	

	block = dim3(outputAmount);
	thread= dim3(256);
	g_NIN_Bgrad<<<block, thread, sizeof(float) * thread.x>>>(curDelta->getDev(),
		bgrad.m_devPoint,
		outputDim,
		outputAmount,
		batch,
		curDelta->getArea());

	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("NIN::getGrad::g_NIN_Bgrad");
}

void NIN::updateWeight()
{
	dim3 block  = outputAmount;
	dim3 thread = min(256, w[0]->getLen());
	g_vecAdd<<<block, thread>>>(momentum_w.m_devPoint, wgrad.m_devPoint, w.m_devPoint,
		momentum_b.m_devPoint, bgrad.m_devPoint, b.m_devPoint,
		w[0]->getLen(), b[0]->getLen(), 
		Config::instance()->getMomentum(),
		Config::instance()->getLrate(), Config::instance()->getLrate());
}

NIN::NIN(std::string name)
{
	m_name = name;
	ConfigNIN* config = (ConfigNIN*)Config::instance()->getLayerByName(m_name);
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
	inputAmount  = preLayer->outputAmount;
	outputAmount = inputAmount;
    //outputAmount = config->m_amount;

	inputDim  = preLayer->outputDim;
	outputDim = inputDim;
	batch     = Config::instance()->getBatchSize();
	lambda    = config->m_weightDecay;
	NON_LINEARITY = config->m_nonLinearity;

	outputs  = new cuMatrix<float>(batch, outputDim * outputDim, outputAmount);
	curDelta = new cuMatrix<float>(batch, outputDim * outputDim, outputAmount);

	for(int i = 0; i < outputAmount; i++){
		w.push_back(new cuMatrix<float>(1, 1, inputAmount));
		b.push_back(new cuMatrix<float>(1, 1, 1));
		wgrad.push_back(new cuMatrix<float>(1, 1, inputAmount));
		bgrad.push_back(new cuMatrix<float>(1, 1, 1));
		wgradTmp.push_back(new cuMatrix<float>(batch, inputAmount, 1));
	}

	w.toGpu();
	b.toGpu();
	wgrad.toGpu();
	bgrad.toGpu();
	wgradTmp.toGpu();

	for(int i = 0; i < outputAmount; i++){
		momentum_w.push_back(new cuMatrix<float>(1, 1, inputAmount));
		momentum_b.push_back(new cuMatrix<float>(1, 1, 1));
	}
	momentum_w.toGpu();
	momentum_b.toGpu();

	this->initRandom();
	Layers::instance()->set(m_name, this);
}

void NIN::save(FILE* file)
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

void NIN::clearMomentum()
{
	for(int i = 0; i < (int)momentum_b.size(); i++){
		momentum_b[i]->gpuClear();
	}
	for(int i = 0; i < (int)momentum_w.size(); i++){
		momentum_w[i]->gpuClear();
	}
}

void NIN::initRandom()
{
	//srand(clock());
	float initW = Config::instance()->getLayerByName(m_name)->m_initW;

	if(Config::instance()->getLayerByName(m_name)->isGaussian()){
		for(int i = 0; i < (int)w.size(); i++){
			float epsilon = initW;
			for(int c = 0; c < w[i]->channels; c++)
			{
				float r1 = 0.5f + 4.0f * (rand()) / RAND_MAX;
				float r2 = 0.5f + 4.0f * (rand()) / RAND_MAX;
				createGaussian(w[i]->getHost() + c * w[i]->getArea(), r1,r2,
					1, 1, w[i]->channels,
					epsilon);
			}
			w[i]->toGpu();
		}
	}
	else{
		for(int i = 0; i < (int)w.size(); i++){
			for(int j = 0; j < (int)w[i]->getLen(); j++){
				w[i]->getHost()[j] =  initW * (2.0f * rand() / RAND_MAX - 1.0f);
				//printf("%f ", w[i]->hostData[j]);
			}//printf("\n");
			w[i]->toGpu();
		}
	}

 		 	
}

void NIN::initFromCheckpoint(FILE* file)
{
	float val = 0;
	for(int a = 0; a < (int)w.size(); a++){
		for(int c = 0; c < w[a]->channels; c++){
			for(int i = 0; i < w[a]->rows; i++){
				for(int j = 0; j < w[a]->cols; j++){
					if(fscanf(file, "%f", &val) == EOF)
                    {
                        LOG("scanf fail", "result/log.txt");
                    }
					w[a]->set(i, j, c, val);
				}
			}
		}

		for(int c = 0; c < b[a]->channels; c++){
			for(int i = 0; i < b[a]->rows; i++){
				for(int j = 0; j < b[a]->cols; j++){
                    if(fscanf(file, "%f", &val) == EOF)
                    {
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
 * dim3 block = dim3(batch, outputAmpunt);
 * dim3 thread= dim3(outputDim * outputDim);
*/

__global__ void g_NIN_feedforward(
	float*  inputs,
	float** ws,
	float** bs,
	float*  outputs,
	int inputDim,
	int outputDim,
	int inputAmount,
	int outputAmount,
	int inputArea,
	int outputArea)
{
	int sp = blockIdx.x;
	int ok = blockIdx.y;
	
	int outputSize2 = outputDim * outputDim;
	int inputSize2 = inputDim* inputDim;

	float  b = bs[ok][0];
    float *w = ws[ok];

	float* curOutput = outputs + ok * outputArea + sp * outputSize2;

	/*convolution*/
	for(int tidx = 0; tidx < outputSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < outputSize2)
		{
			float val = 0.0;
            int skip_add = sp * inputSize2;
			for(int ik = 0; ik < inputAmount; ik++){
				float* curInput = inputs + skip_add;
				val += curInput[idx] * w[ik];
                skip_add += inputArea;
			}
			curOutput[idx] = val + b;
		}
	}
}



/*
 * dim3 block = dim3(batch, inputAmount);
 * dim3 thread= min(inputDim * inputDim, 512);
*/
__global__ void g_NIN_backpropagation(
	float* _curDelta,
	float**ws,
	float* _preDelta,
	int     curDim,
	int     preDim,
	int     preAmount,
	int     curAmount,
	int     curArea,
	int     preArea)
{
    extern __shared__ float wShared[];

	int sp = blockIdx.x;
	int ik = blockIdx.y;

    for(int id = 0; id < curAmount; id += blockDim.x){
        int idx = id + threadIdx.x;
        if(idx < curAmount){
            wShared[idx] = ws[idx][ik];
        }
    }
    __syncthreads();

	int curSize2 = curDim * curDim;
	int preSize2 = preDim * preDim;

	float *preDelta = _preDelta + ik * preArea + sp * preSize2;

	for (int tidx = 0; tidx < preSize2; tidx += blockDim.x) {
		int idx = tidx + threadIdx.x;
		if (idx < preSize2) {
			float val = 0.0;
            int skip_add = sp * curSize2;
			for(int ok = 0; ok < curAmount; ok++){
				float *curDelta = _curDelta + skip_add;
				val += curDelta[idx] * wShared[ok];
                skip_add += curArea;
			}
			preDelta[idx] = val;
		}
	}
}


/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(inputAmount);
*/

__global__ void g_NIN_wgrad(float*_inputs,
	float* _curDelta,
	float** wgradTmp,
	int inputDim,
	int curDeltaDim,
	int inputAmount,
	int outputAmount,
	int inputArea,
	int curDeltaAea)
{
	int ok = blockIdx.y;
	int ik = threadIdx.x;
	int b  = blockIdx.x;

	int inputSize2    = inputDim * inputDim;
	int curDeltaSize2 = curDeltaDim * curDeltaDim;

	float* input    = _inputs +   ik * inputArea   + b * inputSize2;
	float* curDelta = _curDelta + ok * curDeltaAea + b * curDeltaSize2;

	float val = 0.0;
	for(int x = 0; x < inputSize2; x++){
		val += input[x] * curDelta[x];
	}
	wgradTmp[ok][ik + b * inputAmount] = val;
}

/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(THREADS, inputAmount);
*/
template <int INPUTAMOUNT, int THREADS>
__global__ void g_NIN_wgrad_1(float*_inputs,
	float* _curDelta,
	float** wgradTmp,
	int inputDim,
	int curDeltaDim,
	int inputAmount,
	int outputAmount,
	int inputArea,
	int curDeltaAea)
{
	__shared__ float __sum[INPUTAMOUNT][THREADS];
	
	int ok = blockIdx.y;
	int ik = threadIdx.y;
	int b  = blockIdx.x;

	float* _sum = __sum[ik];

	int inputSize2    = inputDim * inputDim;
	int curDeltaSize2 = curDeltaDim * curDeltaDim;

	float* input    = _inputs +   ik * inputArea   + b * inputSize2;
	float* curDelta = _curDelta + ok * curDeltaAea + b * curDeltaSize2;

	float val = 0.0;
	for(int x = 0; x < inputSize2; x += blockDim.x){
		int idx = x + threadIdx.x;
		if(idx < inputSize2){
			val += input[idx] * curDelta[idx];
		}
	}

	_sum[threadIdx.x] = val;
	__syncthreads();
	int len = THREADS;
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

	if(threadIdx.x == 0){
		wgradTmp[ok][ik + b * inputAmount] = _sum[0];
	}
}

/*
 * block = dim3(outputAmount);
 * thread= dim3(256);
 * shared  : sizeof(float) * 256
*/
__global__ void g_NIN_Bgrad(float* delta,
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
