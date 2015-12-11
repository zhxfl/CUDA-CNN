#include "ConvCFM.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
#include "../layers/BranchLayer.h"
#include "../common/util.h"

/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(min(kernelSize * kernelSize, 512), cfm);
*/

__global__ void g_ConvCFM_wgrad_shared(float*_inputs,
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
	cudaStreamSynchronize(0);
	getLastCudaError("ConvCFM:getCost");
}

void ConvCFM::feedforward()
{
	if((inputs == NULL))
	{
		printf("ConvCFM init error\n");
		exit(0);
	}
    int afterPaddingDim = inputDim + padding * 2;
	int sharedMemorySize = sizeof(float) * (afterPaddingDim * afterPaddingDim + inputAmount * kernelSize * kernelSize);
    if(false){
	//if(inputDim * inputDim <= 1024 && checkSharedMemory(0, sharedMemorySize)){
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= dim3(outputDim * outputDim);
        g_ConvCFM_feedforward_shared<<<block, thread, sharedMemorySize, 0>>>(
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
		checkCudaErrors(cudaStreamSynchronize(0));
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
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("convCFM::g_ConvCFM_feedforward");
	}

	if(NON_LINEARITY >= 0){
		dim3 thread = dim3(min(256, outputs->getLen()));
		dim3 block  = dim3(min(256, (outputs->getLen() + thread.x - 1) / thread.x));
		g_nonLinearity<<<block, thread>>>(
			outputs->getDev(), 
			outputs->getLen(),
			NON_LINEARITY);
		checkCudaErrors(cudaStreamSynchronize(0));
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

		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("ConvCFM::g_dnonLinearity");
	}
	
	if(Config::instance()->getLayerByName(m_name)->m_input == std::string("data"))
		return;

    int after_padding_dim = inputDim + kernelSize  - 1;
    size_t sharedMemorySize = sizeof(float)* (after_padding_dim * after_padding_dim + outputAmount * kernelSize * kernelSize);
    if(false){
	//if(inputDim * inputDim <= 1024 && checkSharedMemory(0, sharedMemorySize)){
		dim3 block = dim3(batch, inputAmount);
		dim3 thread= dim3(inputDim * inputDim);
        cudaFuncSetCacheConfig(g_ConvCFM_backpropagation_shared, cudaFuncCachePreferL1);
		g_ConvCFM_backpropagation_shared<<<block, thread, sharedMemorySize>>>(
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
		checkCudaErrors(cudaStreamSynchronize(0));
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
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("ConvCFM::g_ConvCFM_backpropagation");
	}
}

/*
 * block = dim3(outputAmount, kernelSize * kernelSize * cfm);
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
	int kernelSize2 = kernelSize * kernelSize;
	int kid= blockIdx.y % kernelSize2;
    int c = blockIdx.y / kernelSize2;
	int tid = threadIdx.x;
	_sum[tid] = 0;
	__syncthreads();
	int tlen = batch;
	float* wgradTmp = _WgradTmp[ok];
    int skip = c * wgradTmpArea + kid;
	for(int i = 0; i < tlen; i += blockDim.x)
	{
		int b = i + threadIdx.x;
		if(b < tlen)
		{
			_sum[threadIdx.x] += wgradTmp[b * kernelSize2 + skip];
		}
	}
	__syncthreads();
	int len = blockDim.x;
	while(len != 1)
	{
		__syncthreads();
		int skip = (len + 1) >> 1;
		if(tid < skip && (tid + skip < len))
		{
			_sum[tid] += _sum[tid + skip];
		}
        else{
            return;
        }
		len = skip;
	}
	if(tid == 0)
	{
		Wgrad[ok][kid + c * wgradArea] = _sum[0] / batch + w[ok][kid + c * wArea] * lambda;
	}
}

void ConvCFM::getGrad()
{
    int nAfterPadding = inputDim + padding << 1;
    size_t sharedMemorySize = sizeof(float) * (nAfterPadding * nAfterPadding + outputDim * outputDim);
    if(false){
	//if(kernelSize *  kernelSize <= 1024 && checkSharedMemory(0, sharedMemorySize)){
        dim3 block = dim3(batch, outputAmount);
        dim3 thread= dim3(kernelSize * kernelSize);
		g_ConvCFM_wgrad_shared<<<block, thread, sharedMemorySize>>>(
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
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("g_ConvCFM_wgrad_shared");
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

		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("g_ConvCFM_wgrad");
	}
	

	dim3 block  = dim3(outputAmount, kernelSize * kernelSize * cfm);
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
	checkCudaErrors(cudaStreamSynchronize(0));
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

	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ConvCFM::getGrad::g_ConvCFM_Bgrad");
}

void ConvCFM::updateWeight()
{
	dim3 block  = outputAmount;
	dim3 thread = min(256, w[0]->getLen());
	g_vecAdd<<<block, thread, 0, Layers::instance()->get_stream()>>>(momentum_w.m_devPoint, wgrad.m_devPoint, w.m_devPoint,
		momentum_b.m_devPoint, bgrad.m_devPoint, b.m_devPoint,
		w[0]->getLen(), b[0]->getLen(), 
		Config::instance()->getMomentum(),
		Config::instance()->getLrate(), Config::instance()->getLrate());
}

ConvCFM::ConvCFM(std::string name)
{
	m_name = name;
	ConfigConvCFM* config = (ConfigConvCFM*)Config::instance()->getLayerByName(m_name);
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
    float initW = Config::instance()->getLayerByName(m_name)->m_initW;

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
            }
            w[i]->toGpu();
        }
    }


}

void ConvCFM::initFromCheckpoint(FILE* file)
{
    float val = 0;
    for(size_t a = 0; a < w.size(); a++){
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

    extern __shared__ float curInputShared[];
    int outputSize2 = outputDim * outputDim;
    int inputSize2  = inputDim* inputDim;
    int kernelSize2 = kernelSize * kernelSize;

    int after_padding_dim = inputDim + padding * 2;
    int after_padding_dim2 = after_padding_dim * after_padding_dim;

    float *wShared = curInputShared + after_padding_dim2;
    float b = bs[ok][0];
    float* curOutput = outputs + ok * outputArea + sp * outputSize2;

    /*load wShared*/
    float* w = ws[ok];
    int nLen = kernelSize2 * inputAmount;
    for(int li = 0; li < nLen; li += blockDim.x){
        int lix = li + threadIdx.x;
        if(lix < nLen){
            wShared[lix] = w[lix];
        }
    }

    for(int id = 0; id < after_padding_dim2; id += blockDim.x){
        int idx = id + threadIdx.x;
        if(idx < after_padding_dim2){
            curInputShared[idx] = 0;
        }
    }

    __syncthreads();
    int add_padding = after_padding_dim * padding + padding;
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
                float* w = wShared + c * kernelSize2;

                /*load curInputs*/
                for(int li = 0; li < inputSize2; li += blockDim.x){
                    int lix = li + threadIdx.x;
                    if(lix < inputSize2){
                        int _x = lix / inputDim;
                        int _y = lix % inputDim;
                        curInputShared[after_padding_dim * _x + _y + add_padding] = curInput[lix];
                    }
                }
                __syncthreads();

                for(int i = 0; i < kernelSize; i++){
                    int xx = x + i;
                    float* t_curInput = curInputShared + xx * after_padding_dim;
                    float* t_w = w + i * kernelSize;
                    for(int j = 0; j < kernelSize; j++){
                        val += t_curInput[y + j] * t_w[j];
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

    extern __shared__ float curDeltaShared[];
    int after_padding_dim = preDim + kernelSize - 1;
    int after_padding_dim2 = after_padding_dim * after_padding_dim;
    float* wShared = curDeltaShared + after_padding_dim * after_padding_dim;

    float *preDelta = _preDelta + ik * preArea + sp * preSize2;
    int c = ik;	

    int nLen = curAmount * kernelSize2;

    for(int id = 0; id < nLen; id += blockDim.x){
        int idx = id + threadIdx.x;
        if(idx < nLen){
            int _k = idx / kernelSize2;
            int _i = idx % kernelSize2;
            int _x = kernelSize - _i / kernelSize - 1;
            int _y = kernelSize - _i % kernelSize - 1;
            wShared[idx] = ws[_k][c * kernelSize2 + _x * kernelSize + _y];
        }
    }

    for(int id = 0; id < after_padding_dim2; id += blockDim.x){
        int idx = id + threadIdx.x;
        if(idx < after_padding_dim2)
            curDeltaShared[idx] = 0;
    }
    __syncthreads();

    for (int tidx = 0; tidx < preSize2; tidx += blockDim.x) {
        int idx = tidx + threadIdx.x;
        if (idx < preSize2) {
            int i = idx / preDim;
            int j = idx % preDim;
            float val = 0.0;
            int ok = 0;
            while(ok < curAmount){
                float *curDelta = _curDelta + ok * curArea + sp * curSize2;
                float *w = wShared + ok * kernelSize2;

                /*load curDelta*/
                int cur_padding = (after_padding_dim - curDim) / 2;
                for(int li = 0; li < curSize2; li += blockDim.x){
                    int lix = li + threadIdx.x;
                    if(lix < curSize2){
                        int _x = lix / curDim;
                        int _y = lix % curDim;
                        curDeltaShared[(_x + cur_padding) * after_padding_dim + (_y + cur_padding)] = curDelta[lix];
                    }
                }

                __syncthreads();

                for (int x = 0; x < kernelSize; x++) {
                    int cx = i + x;
                    float* t1 = curDeltaShared + cx * after_padding_dim;
                    float* t2 = w + x * kernelSize;
                    for (int y = 0; y < kernelSize; y++) {
                        int cy = j + y;
                        val += t1[cy] * t2[y];
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
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(kernelSize * kernelSize);
 */

__global__ void g_ConvCFM_wgrad_shared(float*_inputs,
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
    extern __shared__ float curDeltaShared[];

    int ok = blockIdx.y;
    int b = blockIdx.x;

    int inputSize2    = inputDim * inputDim;
    int curDeltaSize2 = curDeltaDim * curDeltaDim;
    int kernelSize2   = kernelSize * kernelSize;

    float* inputShared = curDeltaShared + curDeltaSize2;
    float* curDelta = _curDelta + ok * curDeltaAea + b * curDeltaSize2;
    int after_padding_dim = (inputDim + padding << 1);
    int after_padding_dim2 = after_padding_dim * after_padding_dim;
    int add_padding = after_padding_dim * padding + padding;

    for(int id = 0; id < after_padding_dim2; id += blockDim.x){
        int idx = id + threadIdx.x;
        if(idx < after_padding_dim2){
            inputShared[idx] = 0;    
        }
    }
    for(int id = 0; id < curDeltaSize2; id += blockDim.x){
        int idx = id + threadIdx.x;
        if(idx < curDeltaSize2){
            curDeltaShared[idx] = curDelta[idx];        
        }
    }
    __syncthreads();


    for(int c = 0; c < inputAmount; c++){
        int ik = c;
        float* wgrad = wgradTmp[ok] + c * wgradTmpArea + b * kernelSize2;
        float* input = _inputs + ik * inputArea + b * inputSize2;
        for(int id = 0; id < inputSize2; id += blockDim.x){
            int idx = id + threadIdx.x;
            if(idx < inputSize2){
                int _x = idx / inputDim;
                int _y = idx % inputDim;
                inputShared[_x * after_padding_dim + add_padding + _y] = input[idx];
            }
        }

        for(int tidx = 0; tidx < kernelSize2; tidx += blockDim.x)
        {
            int idx = tidx + threadIdx.x;
            if(idx < kernelSize2)
            {
                int i = idx / kernelSize;
                int j = idx % kernelSize;
                float val = 0.0;

                for(int x = 0; x < curDeltaDim; x++){
                    int cx = i + x;
                    float* t1 = inputShared + cx * after_padding_dim;
                    float* t2 = curDeltaShared + x * curDeltaDim;
                    for(int y = 0; y < curDeltaDim; y++)
                    {
                        int cy = j + y;
                        val += t1[cy] * t2[y];
                    }
                }
                wgrad[idx] = val;
            }
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
    int skip = deltaArea * k2;
    for(int i = 0; i < tlen; i += blockDim.x)
    {
        int idx = i + threadIdx.x;
        if(idx < tlen)
        {
            _sum[threadIdx.x] += delta[idx + skip];
        }
    }
    __syncthreads();
    int len = blockDim.x;
    while(len != 1)
    {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < skip && (threadIdx.x + skip < len))
        {
            _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        else{
            return;
        }
        len = skip;
    }
    if(threadIdx.x == 0)
    {
        bgrad[k2][0] = _sum[0] / batch;
    }
}
