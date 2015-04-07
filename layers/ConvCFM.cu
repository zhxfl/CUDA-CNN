#include "ConvCFM.h"
#include "../common/cuBase.h"
#include "../common/Config.h"
/*
*	blocks : dim3(batch, cuKernelScan[0]),
*	threads: dim3(min(convOutputSize * convOutputSize, 512));
*/
__global__ void g_ConvCFM_feedforward_1(
	double** inputs,
	double** ws,
	double** bs,
	double* outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int outputArea,
	int batch,
	int inputAmount,
	int outputAmount,
	int cfm);


/*
	dim3 block = dim3(batch, outputAmount);
	dim3 thread= dim3(4, min(outputDim * outputDim, 256));
	cfm = 1, outputDim , inputDim <=32 ,kernelSize = 5;

*/
__global__ void g_ConvCFM_feedforward_s_cmf1_1(
	double** inputs,
	double** ws,
	double** bs,
	double* outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int outputArea,
	int batch,
	int inputAmount,
	int outputAmount);

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
__global__ void g_ConvCFM_feedforward_2(
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
__global__ void g_ConvCFM_wgrad_2(double* _inputs,
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
* blocks  : dim3(kernelAmount2, kernelSize * kernelSize),
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
* blocks  : dim3(kernelAmount2)
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
* blocks  : dim3(batch, cuKernelScan[cl]),
* threads : dim3(threadidx)
*/
__global__ void g_ConvCFM_wgrad_1(double**_inputs,
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
	int curDeltaAea,
	int batch,
	double lambda);


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
*blocks  : dim3(kernelAmount2)
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
	g_getCost_3<<<dim3(amount), dim3(32), sizeof(double) * 32>>>(cost->getDev(), 
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
		if(inputDim <= 32 && cfm == 1){
		//if(0){
			dim3 block = dim3(batch, outputAmount);
			dim3 thread= dim3(min(outputDim * outputDim, 1024));
			
			g_ConvCFM_feedforward_s_cmf1_1<<<block, thread>>>(
				inputs_1->m_devPoint, w.m_devPoint, b.m_devPoint, outputs->getDev(),
				inputDim, kernelSize, padding, outputDim, outputs->getArea(),batch, inputAmount, outputAmount);
			
			checkCudaErrors(cudaDeviceSynchronize());
			getLastCudaError("convCFM:g_ConvCFM_feedforward_s_cmf1_1");
			
		}else{
			dim3 block = dim3(batch, outputAmount);
			dim3 thread= dim3(min(outputDim * outputDim, 1024));
			g_ConvCFM_feedforward_1<<<block, thread>>>(inputs_1->m_devPoint,
				w.m_devPoint, 
				b.m_devPoint,
				outputs->getDev(),
				inputDim,
				kernelSize,
				padding,
				outputDim,
				outputs->getArea(),
				batch,
				inputAmount,
				outputAmount,
				cfm);
			checkCudaErrors(cudaDeviceSynchronize());
			getLastCudaError("convCFM:g_ConvCFM_feedforward_1");
		}
	}
	else if(inputs_2){
			int remain = min(512 / 128, outputAmount); //32
			int div = (outputAmount + remain - 1) / remain;//1
			dim3 block = dim3(batch, div);
			dim3 thread= dim3(min(outputDim * outputDim, 128), remain);
			/*
				最慢的kernel函数，由于outputDim = 4所以每个block只能分配到16个线程。
			*/
			g_ConvCFM_feedforward_2<<<block, thread>>>(inputs_2->getDev(),
				w.m_devPoint,
				b.m_devPoint,
				outputs->getDev(),
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
			outputs->getDev(), 
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

		g_dnonLinearity<<<block, thread>>>(curDelta->getDev(),
			outputs->getDev(), curDelta->getLen(), NON_LINEARITY);

		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("ConvCFM::g_dnonLinearity");
	}
	
	if(inputs_2){
		dim3 block = dim3(batch, outputAmount);
		dim3 thread= min(outputDim * outputDim, 512);

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


void ConvCFM::getGrad()
{
	if((inputs_1 == NULL && inputs_2 == NULL) || (inputs_1 != NULL && inputs_2 != NULL))
	{
		printf("ConvCFM init error\n");
		exit(0);
	}
	if(inputs_1){
		dim3 block = dim3(cfm, outputAmount);
		dim3 thread= min(kernelSize * kernelSize * batch, 512);

		g_ConvCFM_wgrad_1<<<block, thread>>>(
			inputs_1->m_devPoint,
			curDelta->getDev(),
			wgrad.m_devPoint,
			w.m_devPoint,
			inputDim,
			outputDim,
			kernelSize,
			inputAmount,
			outputAmount,
			padding,
			inputDim * inputDim,
			curDelta->getArea(),
			batch,
			lambda);

		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("g_ConvCFM_wgrad_1");

		block = dim3(amount);
		thread= dim3(256);
		g_ConvCFM_Bgrad_1<<<block,
			thread,
			sizeof(double) * 256>>>
			(curDelta->getDev(),
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
 		dim3 block = dim3(cfm, outputAmount);
 		dim3 thread= min(kernelSize * kernelSize * batch, 512);
 
 		g_ConvCFM_wgrad_2<<<block, thread>>>(inputs_2->getDev(),
 			curDelta->getDev(),
 			wgrad.m_devPoint,
 			w.m_devPoint,
 			inputDim,
 			outputDim,
 			kernelSize,
 			inputAmount,
 			outputAmount,
 			padding,
 			inputs_2->getArea(),
 			curDelta->getArea(),
 			batch,
 			lambda);
 		cudaDeviceSynchronize();
 		getLastCudaError("g_ConvCFM_wgrad_2");

		block = dim3(amount);
		thread= dim3(256);
		g_ConvCFM_Bgrad_2<<<block, thread, sizeof(double) * 256>>>(curDelta->getDev(),
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
		inputAmount = Config::instance()->getChannels();
		amount = config->m_amount;
		outputAmount = amount;
		kernelSize = config->m_kernelSize;
		padding = config->m_padding;

		inputDim  = Config::instance()->getImageSize();
		outputDim = (inputDim - kernelSize + 1) + padding * 2;
		batch     = Config::instance()->getBatchSize();
		lambda = config->m_weightDecay;
		cfm = 1;
		NON_LINEARITY = config->m_nonLinearity;

		outputs  = new cuMatrix<double>(batch, outputDim * outputDim, outputAmount);
		curDelta = new cuMatrix<double>(batch, outputDim * outputDim, outputAmount);
		preDelta = NULL;

		for(int i = 0; i < amount; i++){
			w.push_back(new cuMatrix<double>(kernelSize, kernelSize, cfm));
			b.push_back(new cuMatrix<double>(1, 1, 1));
			wgrad.push_back(new cuMatrix<double>(kernelSize, kernelSize, cfm));
			bgrad.push_back(new cuMatrix<double>(1, 1, 1));
		}
		w.toGpu();
		b.toGpu();
		wgrad.toGpu();
		bgrad.toGpu();

		for(int i = 0; i < amount; i++){
			momentum_w.push_back(new cuMatrix<double>(kernelSize, kernelSize, cfm));
			momentum_b.push_back(new cuMatrix<double>(1, 1, 1));
		}
		momentum_w.toGpu();
		momentum_b.toGpu();
	}
	else {
		ConvLayerBase * preLayer = (ConvLayerBase*)Layers::instance()->get(config->m_input);

		inputs_1 = NULL;
		inputs_2 = preLayer->getOutputs();
		inputAmount = preLayer->outputAmount;
		amount = config->m_amount;
		outputAmount = amount;
		kernelSize = config->m_kernelSize;
		padding = config->m_padding;

		inputDim  = preLayer->outputDim;
		outputDim = (inputDim + 1 - kernelSize) + padding * 2;
		batch     = Config::instance()->getBatchSize();
		lambda    = config->m_weightDecay;
		cfm = config->m_cfm;
		NON_LINEARITY = config->m_nonLinearity;
		
		outputs = new cuMatrix<double> (batch, outputDim * outputDim, outputAmount);
		curDelta = new cuMatrix<double>(batch, outputDim * outputDim, outputAmount);
		preDelta = preLayer->getCurDelta();

		for(int i = 0; i < amount; i++){
			w.push_back(new cuMatrix<double>(kernelSize, kernelSize, cfm));
			b.push_back(new cuMatrix<double>(1, 1, 1));
			wgrad.push_back(new cuMatrix<double>(kernelSize, kernelSize, cfm));
			bgrad.push_back(new cuMatrix<double>(1, 1, 1));
		}

		w.toGpu();
		b.toGpu();
		wgrad.toGpu();
		bgrad.toGpu();

		for(int i = 0; i < amount; i++){
			momentum_w.push_back(new cuMatrix<double>(kernelSize, kernelSize, cfm));
			momentum_b.push_back(new cuMatrix<double>(1, 1, 1));
		}
		momentum_w.toGpu();
		momentum_b.toGpu();
	}

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
				double r1 = 0.01 + 5 * (rand()) / RAND_MAX;
				double r2 = 0.01 + 5 * (rand()) / RAND_MAX;
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

/*
	dim3 block = dim3(batch, outputAmount);
	dim3 thread= dim3(4, min(outputDim * outputDim, 256));
	cfm = 1, outputDim , inputDim <=32 ,kernelSize = 5;

*/
__global__ void g_ConvCFM_feedforward_s_cmf1_1(
	double** inputs,
	double** ws,
	double** bs,
	double* outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int outputArea,
	int batch,
	int inputAmount,
	int outputAmount)
{
	int sp = blockIdx.x;
	int ok = blockIdx.y;

	int outputSize2 = outputDim * outputDim;
	int inputSize2  = inputDim  * inputDim;
	int kernelSize2 = kernelSize* kernelSize;

	double *w = ws[ok];
	//__shared__ double w[5 * 5];

	double* curInput  = inputs[sp] + ok % inputAmount * inputSize2;
	double* curOutput = outputs + ok * outputArea + sp * outputSize2;
	
	/*load the image to shared memory*/
// 	double* _w = ws[ok];
// 	if(threadIdx.x < kernelSize2){
// 		w[threadIdx.x] = _w[threadIdx.x];
// 	}
//	__syncthreads();

	double b = bs[ok][0];

	/*convolution*/
	for(int tidx = 0; tidx < outputSize2; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < outputSize2)
		{
			int x = idx / outputDim;
			int y = idx % outputDim;
			double val = 0.0;
			for(int i = 0; i < kernelSize; i++){
					int xx = x + i - padding;
					int skipx = xx * inputDim;
					int skipk = i  * kernelSize;
				for(int j = 0; j < kernelSize;j++){
					int yy = y + j - padding;

					if(xx >= 0 && xx < inputDim && yy >= 0 && yy < inputDim){

						val += curInput[skipx + yy] * w[skipk + j];
					}
				}
			}
			curOutput[idx] = val + b;
		}
	}
}


/*
	dim3 block = dim3(batch, outputAmount);
	dim3 thread= dim3(min(outputDim * outputDim, 512));
*/
__global__ void g_ConvCFM_feedforward_1(
	double** inputs,
	double** ws,
	double** bs,
	double* outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int outputArea,
	int batch,
	int inputAmount,
	int outputAmount,
	int cfm)
{
	int sp = blockIdx.x;
	int ok  = blockIdx.y;

	int outputSize2 = outputDim * outputDim;
	int inputSize2  = inputDim  * inputDim;
	int kernelSize2 = kernelSize* kernelSize;

	double  b         = bs[ok][0];

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
				double* curInput = inputs[sp] + (ok + c ) % inputAmount * inputSize2;
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
dim3 block = dim3(batch, div);
dim3 thread= dim3(min(outputDim * outputDim, 512, remain));
*/

__global__ void g_ConvCFM_feedforward_2(
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
	int ok  = blockIdx.y * blockDim.y + threadIdx.y;
	if(ok >= outputAmount)return;

	int outputSize2 = outputDim * outputDim;
	int inputSize2  = inputDim* inputDim;
	int kernelSize2 = kernelSize * kernelSize;

	double  b         = bs[ok][0];

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
dim3 block = dim3(batch, outputAmount);
dim3 thread= min(outputDim * outputDim, 512);
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

	for (int tidx = 0; tidx < preSize2; tidx += blockDim.x) {
		int idx = tidx + threadIdx.x;
		if (idx < preSize2) {
			int i = idx / preDim;
			int j = idx % preDim;
			double val = 0.0;

			for(int c = 0; c < cfm; c++){
				int ik = (ok + c) % preAmount;
				double *preDelta = _preDelta + ik * preArea + sp * preSize2;
				double *w = ws[ok] + c * kernelSize2;
				double *curDelta = _curDelta + ok * curArea + sp * curSize2;

				for (int x = 0; x < kernelSize; x++) {
					for (int y = 0; y < kernelSize; y++) {
						int cx = i + x - (kernelSize >> 1);
						int cy = j + y - (kernelSize >> 1);
						int wx = kernelSize - x - 1;
						int wy = kernelSize - y - 1;
						cx -= ((kernelSize >> 1) - padding);
						cy -= ((kernelSize >> 1) - padding);
						if(cx >= 0 && cx < curDim && cy >= 0 && cy < curDim){
							val += curDelta[cx * curDim + cy] * w[wx * kernelSize + wy];
						}
					}
				}
				atomicAdd(preDelta + idx, val);
			}
			//atomicAdd(preDelta + idx, val);
		}
	}
}


/*
dim3 block = dim3(cfm, outputAmount);
dim3 thread= min(kernelSize * kernelSize * batch, 512);
*/

__global__ void g_ConvCFM_wgrad_2(double* _inputs,
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
	double lambda)
{
	int c  = blockIdx.x;
	int ok = blockIdx.y;
	int ik = (ok + c) % inputAmount;

	int inputSize2    = inputDim * inputDim;
	int curDeltaSize2 = curDeltaDim * curDeltaDim;
	int kernelSize2   = kernelSize * kernelSize;

	double* wgrad   = _wgrad[ok] + 
		+ c * kernelSize2;

	int cfm = gridDim.x;
	int len = kernelSize2 * cfm;
	for(int i = 0; i < len; i += blockDim.x){
		int idx = i + threadIdx.x;
		if(idx < len){
			_wgrad[ok][idx] = 0;
		}
	}

	__syncthreads();

	len = kernelSize2 * batch;
	for(int tidx = 0; tidx < len; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < len)
		{
			int b = idx / kernelSize2;
			int t = idx % kernelSize2;
			int i = t   / kernelSize;
			int j = t   % kernelSize;
			double val = 0.0;

			double* input    = _inputs   + ik * inputArea    + b * inputSize2;
			double* curDelta = _curDelta + ok * curDeltaArea + b * curDeltaSize2;

			for(int x = 0; x < curDeltaDim; x++){
				int cx = i + x - padding;
				for(int y = 0; y < curDeltaDim; y++)
				{
					int cy = j + y - padding;
					if(cx >= 0 &&  cy >= 0 && cx < inputDim && cy < inputDim)
						val += input[cx * inputDim + cy] * curDelta[x * curDeltaDim + y];
				}
			}
			atomicAdd(wgrad + t, val);
		}
	}

	__syncthreads();
	for(int i = 0; i < kernelSize2; i += blockDim.x){
		int idx = i + threadIdx.x;
		if(idx < kernelSize2){
			wgrad[idx] = wgrad[idx] / batch + w[ok][c * kernelSize2 + idx] * lambda ;
		}
	}
}




/*
dim3 block = dim3(inputAmount, outputAmount, kernelSize * kernelSize);
dim3 thread= min(batch, 256);

g_ConvCFM_wgrad_2<<<block, thread>>>(inputs_2->getDev(),
curDelta->getDev(),
wgrad.m_devPoint,
w.m_devPoint,
inputDim,
outputDim,
kernelSize,
inputAmount,
outputAmount,
padding,
inputs->getArea(),
curDelta->getArea(),
batch,
lambda);
cudaDeviceSynchronize();
*/

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
	double lambda)
{
	extern __shared__ double _sum[]; 
	int ik  = blockIdx.x;
	int ok  = blockIdx.y;
	int idx = blockIdx.z;

	int inputSize2    = inputDim * inputDim;
	int curDeltaSize2 = curDeltaDim * curDeltaDim;
	int kernelSize2   = kernelSize * kernelSize;

	double* wgrad   = _wgrad[ok] + 
		+ ik * kernelSize2;

	int len = batch;
	for(int tidx = 0; tidx < len; tidx += blockDim.x)
	{
		int b = tidx + threadIdx.x;
		if(idx < len)
		{
			int i = idx / kernelSize;
			int j = idx % kernelSize;
			double val = 0.0;

				double* input    = _inputs   + ik * inputArea    + b * inputSize2;
				double* curDelta = _curDelta + ok * curDeltaArea + b * curDeltaSize2;

				for(int x = 0; x < curDeltaDim; x++){
					for(int y = 0; y < curDeltaDim; y++)
					{
						int cx = i + x - padding;
						int cy = j + y - padding;
						if(cx >= 0 &&  cy >= 0 && cx < inputDim && cy < inputDim)
							val += input[cx * inputDim + cy] * curDelta[x * curDeltaDim + y];
					}
				}
			_sum[b] = val;
		}
	}

	__syncthreads();
	len = batch;
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

	if(threadIdx.x == 0){
		wgrad[idx] = _sum[0] / batch + w[ok][ik * kernelSize2 + idx] * lambda;
	}
}
/*
* blocks  : dim3(kernelAmount2, kernelSize * kernelSize),
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
* blocks  : dim3(kernelAmount2)
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
/*

g_ConvCFM_wgrad_1<<<block, thread>>>(
double**_inputs,
double* _curDelta,
double* _wgrad,
int inputDim,
int curDeltaDim,
int kernelSize,

int inputAmount,
int outputAmount,
int padding,

int inputArea,
int curDeltaAea,
int wgrapArea,
int batch)
dim3 block = dim3(cfm, outputAmount);
dim3 thread= min(kernelSize * kernelSize * batch, 512);
*/
__global__ void g_ConvCFM_wgrad_1(
	double**_inputs,
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
	int curDeltaAea,
	int batch,
	double lambda)
{
	int ok = blockIdx.y;
	int ik = (blockIdx.x + ok) % inputAmount;
	int c  = blockIdx.x;

	int inputSize2    = inputDim * inputDim;
	int curDeltaSize2 = curDeltaDim * curDeltaDim;
	int kernelSize2   = kernelSize * kernelSize;

	double* wgrad   = _wgrad[ok] + c * kernelSize2;

	int cfm = gridDim.x;
	int len = kernelSize2 * cfm;
	for(int i = 0; i < len; i += blockDim.x){
		int idx = i + threadIdx.x;
		if(idx < len){
			_wgrad[ok][idx] = 0;
		}
	}

	__syncthreads();

	len = kernelSize2 * batch;

	for(int tidx = 0; tidx < len; tidx += blockDim.x)
	{
		int idx = tidx + threadIdx.x;
		if(idx < len)
		{
			int b = idx / kernelSize2;
			int t = idx % kernelSize2;
			int i = t   / kernelSize;
			int j = t   % kernelSize;
			double val = 0.0;

			double* input    = _inputs[b]+ ik * inputSize2;
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

			atomicAdd(wgrad + t, val);
		}
	}

	__syncthreads();

	for(int i = 0; i < kernelSize2; i += blockDim.x){
		int idx = i + threadIdx.x;
		if(idx < kernelSize2){
			wgrad[idx] = wgrad[idx] / batch + w[ok][c * kernelSize2 + idx] * lambda ;
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
*blocks  : dim3(kernelAmount2)
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