#include "cuTrasformation.cuh"
#include <math.h>
#include <stdio.h>
#include "../common/cuMatrix.h"
#include "../common/util.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <time.h>
#include "../common/Config.h"
#include <helper_functions.h>
#include <helper_cuda.h>
#include "../common/cuBase.h"


#define GAUSSIAN_FIELD_SIZE (21) /* strictly odd number */
#define constDistortion (1.0)
curandGenerator_t rand_generator_device;
const curandRngType_t generator_type = CURAND_RNG_PSEUDO_DEFAULT;

cuMatrix<float>* cuGaussianKernel;
cuMatrix<float>* cuDispH;
cuMatrix<float>* cuDispV;

float * cu_d_randonNumf;
float* cu_d_randomNum;
float* cu_h_randomNum;
float dElasticSigma   = 4.0;   /* higher numbers are more smooth and less distorted; Simard uses 4.0*/


int getRandomNumLen(int batch, int ImgSize)
{
	return batch * ImgSize * ImgSize * 2 * Config::instance()->getChannels();
}

/*
 * blocks : dim3(1)
 * threads: dim3(GAUSSIAN_FIELD_SIZE*GAUSSIAN_FIELD_SIZE)
*/
__global__ void g_createGaussianKernel(float* gaussian, float dElasticSigma, int ImgSize)
{
	int iiMid = GAUSSIAN_FIELD_SIZE >> 1;
	float floatElasticSigma = dElasticSigma * dElasticSigma;
	int row = threadIdx.x % ImgSize;
	int col = threadIdx.x / ImgSize;
	float val1 = 1.0 / (dElasticSigma * 2.0 * 3.1415926535897932384626433832795);
	float val2 = (row-iiMid)*(row-iiMid) + (col-iiMid)*(col-iiMid);

	gaussian[threadIdx.x] = val1 * exp(-1.0 * val2 / (2.0 * floatElasticSigma));
}

void cuInitDistortionMemery(int batch, int ImgSize)
{
	curandStatus_t curandstatus;
	cuGaussianKernel = new cuMatrix<float>(GAUSSIAN_FIELD_SIZE, GAUSSIAN_FIELD_SIZE, 1);
	if(GAUSSIAN_FIELD_SIZE * GAUSSIAN_FIELD_SIZE > MAX_THREADS)
	{
		char logStr[1024];
		sprintf(logStr, "g_createGaussianKernel > MAX_THREADS\n");
		LOG(logStr, "Result/log.txt");
		exit(0);
	}
	g_createGaussianKernel<<<dim3(1),dim3(GAUSSIAN_FIELD_SIZE * GAUSSIAN_FIELD_SIZE)>>>(
		cuGaussianKernel->getDev(),
		dElasticSigma,
		ImgSize);
	cudaDeviceSynchronize();

	/*cu_d_randomNum*/
	checkCudaErrors(
		MemoryMonitor::instance()->gpuMalloc((void**)&cu_d_randomNum, sizeof(float) * getRandomNumLen(batch, ImgSize))
		);
	/*cu_d_randonNumf*/
	checkCudaErrors(
		MemoryMonitor::instance()->gpuMalloc((void**)&cu_d_randonNumf, sizeof(float) * getRandomNumLen(batch, ImgSize))
		);
	/*cu_h_randomNum*/
	cu_h_randomNum = (float*)MemoryMonitor::instance()->cpuMalloc(sizeof(float) * getRandomNumLen(batch, ImgSize));
	if(!cu_h_randomNum)
	{
		char logStr[1024];
		sprintf(logStr, "malloc cu_h_randomNum fail\n");
		LOG(logStr, "Result/log.txt");
		exit(0);
	}

	/*curandCreateGenerator*/
	curandstatus = curandCreateGenerator(&rand_generator_device, generator_type);
	if(curandstatus != CURAND_STATUS_SUCCESS)
	{
		char logStr[1024];
		sprintf(logStr, "curandCreateGenerator fail\n");
		LOG(logStr, "Result/log.txt");
		exit(0);
	}

	cuDispV = new cuMatrix<float>(batch, ImgSize * ImgSize, 1);
	cuDispH = new cuMatrix<float>(batch, ImgSize * ImgSize, 1);
}


__global__ void g_getRandomUniform(float* r1, float* r2, int len)
{
	for(int i = 0; i < len; i += gridDim.x * blockDim.x)
	{
		int id = i + blockDim.x * blockIdx.x + threadIdx.x;
		if(id < len)
		{
			r2[id] = r1[id] * 2.0f - 1.0f;
		}
	}
}

/*
 * blocks  : dim3(batch)
 * threads : dim3(512)
 */
__global__ void g_generateDistortionMap(
	float* _dispH,
	float* _dispV,
	float* rand, 
	float* gaussianKernel,
	float dElasticScaling, 
	float dMaxScaling,
	float dMaxRotation,
	int ImgSize)
{
	int ImgSize2 = ImgSize * ImgSize;

	float* uniformH = rand + blockIdx.x * ImgSize2;
	float* uniformV = rand + blockIdx.x * ImgSize2 * 2;
	float* dispH = _dispH + ImgSize2 * blockIdx.x;
	float* dispV = _dispV + ImgSize2 * blockIdx.x;

	if(dElasticScaling >= 0.1){
		for(int is = 0; is < ImgSize2; is += blockDim.x)
		{
			int idx = is + threadIdx.x;
			if(idx < ImgSize2)
			{
				int row = idx / ImgSize;
				int col = idx % ImgSize;

				int iiMid = GAUSSIAN_FIELD_SIZE / 2;

				float fConvolvedH = 0.0;
				float fConvolvedV = 0.0;
				float fSampleH, fSampleV;

				float elasticScale = dElasticScaling;

				for(int xxx = 0; xxx < GAUSSIAN_FIELD_SIZE; ++xxx)
				{
					for(int yyy = 0; yyy < GAUSSIAN_FIELD_SIZE; ++yyy)
					{
						int xxxDisp = col - iiMid + xxx;
						int yyyDisp = row - iiMid + yyy;

						if(xxxDisp < 0 || xxxDisp >= ImgSize || 
							yyyDisp < 0 || yyyDisp >= ImgSize)
						{
							fSampleH = 0.0;
							fSampleV = 0.0;
						}
						else 
						{
							fSampleH = uniformH[yyyDisp * ImgSize + xxxDisp];
							fSampleV = uniformV[yyyDisp * ImgSize + xxxDisp];
						}


						fConvolvedH += fSampleH * gaussianKernel[yyy * GAUSSIAN_FIELD_SIZE + xxx] * constDistortion;
						fConvolvedV += fSampleV * gaussianKernel[yyy * GAUSSIAN_FIELD_SIZE + xxx] * constDistortion;
					}
				}

				dispH[idx] = elasticScale * fConvolvedH;
				dispV[idx] = elasticScale * fConvolvedV;
			}
		}
	}
	else{
		for(int is = 0; is < ImgSize2; is += blockDim.x){
			int idx = is + threadIdx.x;
			if(idx < ImgSize2){
				dispH[idx] = 0.0;
				dispV[idx] = 0.0;
			}
		}
	}
	__syncthreads();

	float rand1 = rand[blockIdx.x];
	float rand2 = rand[blockIdx.x];
	if(fabs(dMaxRotation) >= 0.01){
		rand1 += 1.0;
		rand2 += 1.0;
	}

	for(int is = 0; is < ImgSize2; is += blockDim.x)
	{
		int idx = is + threadIdx.x;
		if(idx < ImgSize2)
		{
			int row = idx / ImgSize;
			int col = idx % ImgSize;

			float dSFHoriz = dMaxScaling / 100.0 * rand1;
			float dSFVert  = dMaxScaling / 100.0 * rand2;

			int iMid = ImgSize / 2;

			dispH[idx] += dSFHoriz * (col - iMid);
			dispV[idx] += dSFVert  * (row - iMid);

		}
	}
	__syncthreads();

	for(int is = 0; is < ImgSize2; is += blockDim.x)
	{
		int idx = is + threadIdx.x;
		if(idx < ImgSize2)
		{
			int row = idx / ImgSize;
			int col = idx % ImgSize;
			float angle = dMaxRotation * rand[blockIdx.x];
			//printf("%f\n",angle);
			angle = angle * 3.1415926535897932384626433832795 / 180.0;

			float cosAngle = cos(angle);
			float sinAngle = sin(angle);

			int iMid = ImgSize / 2;

			float xx = row - iMid;
			float yy = col - iMid;

			dispH[idx] += yy - yy * cosAngle - xx * sinAngle;
			dispV[idx] += xx - xx * cosAngle + yy * sinAngle;
		}
	}
}


__global__ void g_scaleAndRotate(
	float* _dispH,
	float* _dispV,
	float scalingx,
	float scalingy,
	float rotation,
	int ImgSize)
{
	int ImgSize2 = ImgSize * ImgSize;

	float* dispH = _dispH + ImgSize2 * blockIdx.x;
	float* dispV = _dispV + ImgSize2 * blockIdx.x;

	for(int is = 0; is < ImgSize2; is += blockDim.x)
	{
		int idx = is + threadIdx.x;
		if(idx < ImgSize2)
		{
			dispH[idx] = 0.0;
			dispV[idx] = 0.0;
		}
	}
	__syncthreads();

	for(int is = 0; is < ImgSize2; is += blockDim.x)
	{
		int idx = is + threadIdx.x;
		if(idx < ImgSize2)
		{
			int row = idx / ImgSize;
			int col = idx % ImgSize;
			float dSFHoriz = scalingx / 100.0;
			float dSFVert  = scalingy / 100.0;

			int iMid = ImgSize / 2;

			dispH[idx] += dSFHoriz * (col - iMid);
			dispV[idx] += dSFVert  * (row - iMid);
		}
	}
	__syncthreads();

	for(int is = 0; is < ImgSize2; is += blockDim.x)
	{
		int idx = is + threadIdx.x;
		if(idx < ImgSize2)
		{
			int row = idx / ImgSize;
			int col = idx % ImgSize;
			float angle = rotation;
			angle = angle * 3.1415926535897932384626433832795 / 180.0;

			float cosAngle = cos(angle);
			float sinAngle = sin(angle);

			int iMid = ImgSize / 2;

			float xx = row - iMid;
			float yy = col - iMid;

			dispH[idx] += yy - yy * cosAngle - xx * sinAngle;
			dispV[idx] += xx - xx * cosAngle + yy * sinAngle;
		}
	}
}

/*
 * blocks : dim3(batch, Config::instance()->getChannels())
 * threads: dim3(min(512, ImgSize * ImgSize))
 */
__global__ void g_applyDistortionMap(
	float** _inputs,
	float** _outputs,
	float* _dispH, 
	float* _dispV, 
	int ImgSize)
{

	extern __shared__ float img[];
	int c = blockIdx.y;

	int ImgSize2 = ImgSize * ImgSize;
	float* input = _inputs[blockIdx.x] + ImgSize2 * c;
	float* output= _outputs[blockIdx.x]+ ImgSize2 * c;
	float* dispV = _dispV + blockIdx.x * ImgSize2;
	float* dispH = _dispH + blockIdx.x * ImgSize2;

	for(int is = 0; is < ImgSize2; is += blockDim.x){
		int idx = is + threadIdx.x;
		if(idx < ImgSize2){
			img[idx] = input[idx];
		}
	}

	__syncthreads();

	for(int is = 0; is < ImgSize2; is += blockDim.x)
	{
		int idx = is + threadIdx.x;
		if(idx < ImgSize2)
		{
			int row = idx / ImgSize;
			int col = idx % ImgSize;

			float sourceRow, sourceCol;
			float fracRow, fracCol;
			float w1, w2, w3, w4;
			float sourceValue;
			int sRow, sCol, sRowp1, sColp1;
			bool bSkipOutOfBounds;

			if(fabs(dispV[idx]) < 0.000000001 && fabs(dispH[idx]) < 0.0000000001)
			{
				output[idx] = input[idx];
				continue;
			}
			sourceRow = (float)row - dispV[idx];
			sourceCol = (float)col - dispH[idx];

			fracRow = sourceRow - (int)sourceRow;
			fracCol = sourceCol - (int)sourceCol;

			w1 = ( 1.0 - fracRow ) * ( 1.0 - fracCol );
			w2 = ( 1.0 - fracRow ) * fracCol;
			w3 = fracRow * ( 1.0 - fracCol );
			w4 = fracRow * fracCol;

			bSkipOutOfBounds = false;

			if ( ((int)sourceRow + 1) >= ImgSize )	bSkipOutOfBounds = true;
			if ( (int)sourceRow < 0 )				bSkipOutOfBounds = true;

			if ( ((int)sourceCol + 1) >= ImgSize )	bSkipOutOfBounds = true;
			if ( (int)sourceCol < 0 )				bSkipOutOfBounds = true;

			if ( bSkipOutOfBounds == false )
			{
				sRow = (int)sourceRow;
				sCol = (int)sourceCol;

				sRowp1 = sRow + 1;
				sColp1 = sCol + 1;

				while (sRowp1 >= ImgSize) sRowp1 -= ImgSize;
				while (sRowp1 < 0) sRowp1 += ImgSize;

				while (sColp1 >= ImgSize) sColp1 -= ImgSize;
				while (sColp1 < 0) sColp1 += ImgSize;

				while (sRow >= ImgSize) sRow -= ImgSize;
				while (sRow < 0) sRow += ImgSize;

				while (sCol >= ImgSize) sCol -= ImgSize;
				while (sCol < 0) sCol += ImgSize;

				sourceValue =	
					w1 * img[sRow   * ImgSize + sCol] +
					w2 * img[sRow   * ImgSize + sColp1] +
					w3 * img[sRowp1 * ImgSize + sCol] +
					w4 * img[sRowp1 * ImgSize + sColp1];
			}
			else
			{
				sourceValue = -1.0;  
			}
			output[idx] = sourceValue;
		}
	}
}

void cuApplyRandom(int batch, unsigned long long s, int ImgSize)
{
	curandStatus_t curandStatus;
	unsigned long long seed = s;
	curandStatus = curandSetPseudoRandomGeneratorSeed(rand_generator_device, seed);
	if(curandStatus != CURAND_STATUS_SUCCESS)
	{
		char logStr[1024];
		sprintf(logStr, "curandSetPseudoRandomGeneratorSeed fail\n");
		LOG(logStr, "Result/log.txt");
		exit(0);
	}

	curandGenerateUniform(rand_generator_device, cu_d_randonNumf, getRandomNumLen(batch, ImgSize));

	g_getRandomUniform<<<dim3(256),dim3(256)>>>(cu_d_randonNumf, cu_d_randomNum, getRandomNumLen(batch, ImgSize));
	cudaDeviceSynchronize();
	getLastCudaError("g_getRandomUniform");

	int threads = min(512, ImgSize * ImgSize);
	g_generateDistortionMap<<<dim3(batch),threads>>>(cuDispH->getDev(),
		cuDispV->getDev(), cu_d_randomNum, cuGaussianKernel->getDev(),
		Config::instance()->getDistortion(),
		Config::instance()->getScale(),
		Config::instance()->getRotation(), ImgSize);
	cudaDeviceSynchronize();
	getLastCudaError("g_generateDistortionMap");
}

void cuApplyScaleAndRotate(int batch,
		int ImgSize,
		float scalingx,
		float scalingy,
		float rotation)
{
	g_scaleAndRotate<<<dim3(batch),dim3(512)>>>(
			cuDispH->getDev(),
			cuDispV->getDev(),
			scalingx,
			scalingy,
			rotation,
			ImgSize);
	cudaDeviceSynchronize();
	getLastCudaError("g_generateDistortionMap");

}

void cuApplyDistortion(float**inputs, float**outputs, int batch, int ImgSize)
{
	int threadidx = min(ImgSize * ImgSize, 512);
	g_applyDistortionMap<<<dim3(batch, Config::instance()->getChannels()),
		dim3(threadidx), sizeof(float) * ImgSize * ImgSize>>>(inputs,
		outputs, 
		cuDispH->getDev(),
		cuDispV->getDev(),
		ImgSize);
	cudaDeviceSynchronize();
	getLastCudaError("g_applyDistortionMap");
}

/*
 * blocks  : dim3(batch, channels)
 * threads : dim3(min(ImgSize*ImgSize, 512))
 */
__global__ void g_applyCropRandom(float**_inputs, float**_outputs, float* random, int crop, int ImgSize)
{
	int c = blockIdx.y;

	int outputImgSize = ImgSize;
	int inputImgSize  = ImgSize + crop;

	int inputImgSize2 = inputImgSize * inputImgSize;
	int outputImgSize2= outputImgSize* outputImgSize;

	float* input = _inputs [blockIdx.x] + c * inputImgSize2;
	float* output= _outputs[blockIdx.x] + c * outputImgSize2;

	int sx =(int)((random[blockIdx.x]     + 1.0) * 0.5 * crop);
	int sy =(int)((random[blockIdx.x + 1] + 1.0) * 0.5 * crop);

	if(sx > crop) sx = crop;
	if(sy > crop) sy = crop;

	if(sx < 0) sx = 0;
	if(sy < 0) sy = 0;
// 	if(threadIdx.x == 0)
// 		sprintf(logStr, "%d %d\n", sx, sy);

	for(int is = 0; is < outputImgSize2; is += blockDim.x)
	{
		int idx = is + threadIdx.x;
		if(idx < outputImgSize2)
		{
			int ox  = idx / outputImgSize;
			int oy  = idx % outputImgSize;

			int ix  = ox + sx;
			int iy  = oy + sy;
			
			cuAssert(ix < inputImgSize && iy < inputImgSize);
			output[idx] = input[ix * inputImgSize + iy];
		}
	}
}



/*
 * blocks : dim3(batch, channels)
 * threads: dim3(min(ImgSize * ImgSize, 512);
*/
__global__ void g_applyCrop(float**_inputs, float**_outputs, float* random, int croplen, int ImgSize, int cropr, int cropc)
{
	int c = blockIdx.y;
	
	int outputImgSize = ImgSize;
	int inputImgSize  = ImgSize + croplen;

	int inputImgSize2 = inputImgSize * inputImgSize;
	int outputImgSize2= outputImgSize* outputImgSize;

	float* input = _inputs [blockIdx.x]+ c * inputImgSize2 ;
	float* output= _outputs[blockIdx.x]+ c * outputImgSize2;

	int sx = cropr;
	int sy = cropc;

	for(int is = 0; is < outputImgSize2; is += blockDim.x)
	{
		int idx = is + threadIdx.x;
		if(idx < outputImgSize2)
		{
			int ox  = idx / outputImgSize;
			int oy  = idx % outputImgSize;
			int ix  = ox + sx;
			int iy  = oy + sy;
			cuAssert(ix < inputImgSize && iy < inputImgSize);
			output[idx] = input[ix * inputImgSize + iy];
		}
	}
}

void cuApplyCropRandom(float**inputs, float**outputs, int batch, int ImgSize)
{
	dim3 block = dim3(batch, Config::instance()->getChannels());

	dim3 threads = min(512, ImgSize * ImgSize);

	g_applyCropRandom<<<block,threads>>>(inputs, outputs, cu_d_randomNum, Config::instance()->getCrop(), ImgSize);
	cudaDeviceSynchronize();
	getLastCudaError("g_applyCropRandom");
}

void cuApplyCrop(float**inputs, float**outputs, int batch, int ImgSize, int cropr, int cropc)
{
	int threads = min(512, ImgSize * ImgSize);
	g_applyCrop<<<dim3(batch, Config::instance()->getChannels()),
		dim3(threads)>>>(inputs, outputs,cu_d_randomNum, Config::instance()->getCrop(), ImgSize, cropr, cropc);
	cudaDeviceSynchronize();
	getLastCudaError("g_applyCrop");
}

/*
 * function: orizontal Reflection
 * blocks  : dim3(batch, Config::instance()->getChannels()),
 * threads : dim3(threads)
 * flag : 0. Random
 *		  1. Horizontal
 *		  2. Not Horizontal
 */
__global__ void g_applyHorizontal(float**_inputs, float**_outputs, float* rand, int ImgSize, int flag)
{
	int c = blockIdx.y;

	int ImgSize2 = ImgSize * ImgSize ;

	float* input = _inputs[blockIdx.x] + c * ImgSize2;
	float* output= _outputs[blockIdx.x]+ c * ImgSize2;

	int half = ImgSize / 2;
	for(int is = 0; is < half * ImgSize; is += blockDim.x)
	{
		int idx = is + threadIdx.x;
		if(idx < half * ImgSize)
		{
			int ox = idx / half;
			int oy = idx % half;
			int ix = ox;
			int iy = ImgSize - oy - 1;
			if(flag == RANDOM_HORIZONTAL)
			{
				//if(rand[blockIdx.x] <= 0.0){
				if(blockIdx.x % 2 == 0){
					cuAssert(ix < ImgSize && iy < ImgSize);
					swap(output[ox * ImgSize + oy], input[ix * ImgSize + iy]);
				}
			}
			else if(flag == HORIZONTAL){
				cuAssert(ix < ImgSize && iy < ImgSize);
				swap(output[ox * ImgSize + oy], input[ix * ImgSize + iy]);
			}
			else if(flag == NOT_HORIZONTAL){
			}
		}
	}
}

/*
 * flag : 0. Random
 *		  1. Horizontal
 *		  2. Not Horizontal
*/
void cuApplyHorizontal(float **inputs, float**outputs, int batch, int ImgSize, int flag)
{
	int threads = std::min(ImgSize * ImgSize / 2, 512);

	g_applyHorizontal<<<dim3(batch, Config::instance()->getChannels()),
		dim3(threads)>>>(inputs, outputs, cu_d_randomNum,  ImgSize, flag);

	cudaDeviceSynchronize();
	getLastCudaError("g_applyHorizontal");
}


__global__ void g_applyWhiteNoise(
	float** _inputs, 
	float ** _outputs,
	float * _random,
	int ImgSize, 
	float stdev){

		int s = blockIdx.x;
		int c = blockIdx.y;

		int ImgSize2 = ImgSize * ImgSize;

		int offset = ImgSize2 * c;
		float* input = _inputs [s] + offset;
		float* output= _outputs[s] + offset;

		float* rand = _random + offset;
		//if(_random[blockIdx.x] >= 0.9){
		if(true){
			for(int i = 0; i < ImgSize2; i += blockDim.x){
				int idx = i + threadIdx.x;
				if(idx < ImgSize2){
					float val = input[idx] + stdev * rand[idx];
// 					if(val < -1.0) val = -1.0;
// 					if(val >  1.0) val = 1.0;
					output[idx] = val;
				}
			}
		}else{
			for(int i = 0; i < ImgSize2; i += blockDim.x){
				int idx = i + threadIdx.x;
				if(idx < ImgSize2){
					output[idx] = input[idx];
				}
			}
		}
}


/*
ref: http://en.wikipedia.org/wiki/White_noise
*/
void cuApplyWhiteNoise(float **inputs, float**outputs, int batch, int ImgSize, float stdev)
{
	dim3 blocks  = dim3(batch, Config::instance()->getChannels());
	dim3 threads = dim3(min(ImgSize * ImgSize, 512));
	
	g_applyWhiteNoise<<<blocks, threads>>>(inputs, outputs, cu_d_randomNum, ImgSize, stdev);
	cudaDeviceSynchronize();
	getLastCudaError("g_applyWhiteNoise");
}
