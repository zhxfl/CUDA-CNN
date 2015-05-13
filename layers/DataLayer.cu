#include "DataLayer.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
#include "../common/Config.h"
#include "../common/cuBase.h"
#include "../common/util.h"
#include "../dataAugmentation/cuTrasformation.cuh"

using namespace cv;

/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(min(outputDim * outputDim, 1024));
*/
__global__ void g_dataLayer_feedforward(
	float** inputs,
	float* outputs,
	int outputArea,
	int outputCols);

DataLayer::DataLayer(std::string name){
	m_name = name;
	batchId = 0;

	inputDim  = Config::instance()->getImageSize() + Config::instance()->getCrop();
	outputDim = Config::instance()->getImageSize();
	batch     = Config::instance()->getBatchSize();
	inputAmount = Config::instance()->getChannels();
	outputAmount= inputAmount;
	outputs = new cuMatrix<float>(batch, outputDim * outputDim, outputAmount);
	color_noise = new cuMatrix<float>(batch, 3, 1);


	for(int i = 0; i < 2; i ++){
		for(int j = 0; j < batch; j++){
			batchImg[i].push_back(new cuMatrix<float>(inputDim, inputDim, Config::instance()->getChannels()));
		}
		batchImg[i].toGpu();
	}

	for(int i = 0; i < batch; i++){
		cropOutputs.push_back(new cuMatrix<float>(outputDim, outputDim, Config::instance()->getChannels()));
		cropOutputs.toGpu();
	}

	checkCudaErrors(cudaStreamCreate(&stream1));

	/*load PCA*/
	readEigen(eigenValues, eigenVectors);
	printf("read eigen\n");
	Layers::instance()->set(m_name, this);
}

/*
 * dim3 block = dim3(batch, outputAmount);
 * dim3 thread= dim3(min(outputDim * outputDim, 1024));
*/

__global__ void g_dataLayer_feedforward(
	float** inputs,
	float* outputs,
	int outputArea,
	int outputCols)
{
	int batchId = blockIdx.x;
	int ok      = blockIdx.y;

	float* input = inputs[batchId];
	float* output= outputs + ok * outputArea + batchId * outputCols;
	for(int i = 0; i < outputCols; i += blockDim.x){
		int idx = i + threadIdx.x;
		if(idx < outputCols){
			output[idx] = input[idx];
		}
	}
}

/*distortion the data*/
void DataLayer::feedforward(){
	dim3 block = dim3(batch, outputAmount);
	dim3 thread= dim3(min(outputDim * outputDim, 1024));
	
	g_dataLayer_feedforward<<<block, thread>>>(
		cropOutputs.m_devPoint, 
		outputs->getDev(),
		outputs->getArea(),
		outputs->cols);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("DataLayer:feedforward");
	
}; 

void DataLayer::readEigen(Mat& eigenValues, Mat& eigenVecotrs){
	eigenValues  = Mat::zeros(3, 1, CV_32FC1);
	eigenVecotrs = Mat::zeros(3, 3, CV_32FC1);

	FILE *pOut = fopen("eigen", "r");

	for(int i = 0; i < eigenValues.rows; i++){
		fscanf(pOut, "%f", &eigenValues.at<float>(i));
	}

	for(int i = 0; i < eigenVecotrs.rows; i++){
		for(int j = 0; j < eigenVecotrs.cols; j++){
			fscanf(pOut, "%f", &eigenVecotrs.at<float>(i, j));
		}
	}
	fclose(pOut);
}

void DataLayer::trainData()
{
	/*add noise*/
	if(Config::instance()->getChannels() == 3){
		Mat matrix2xN(batch, 3, CV_32FC1);
		randn(matrix2xN, 0, 0.1f);
		for(int i = 0; i < batch; i++){
			Mat	nev(3, 1, CV_32FC1);
			eigenValues.copyTo(nev);

			for(int j = 0; j < 3; j++){
				nev.at<float>(j) = matrix2xN.at<float>(i, j) * eigenValues.at<float>(j);
			}

			Mat t = nev.t() * eigenVectors.t();
			for(int j = 0; j < 3; j++){
				color_noise->set(i, j, 0,  t.at<float>(j));
				printf("%f ", t.at<float>(j));
			}printf("\n");
		}
		color_noise->toGpu();
		cuApplyColorNoise(batchImg[batchId].m_devPoint, color_noise->getDev(), batch, inputDim);
	}
	
// 		showImg(img, 5);
// 
//  		int key = 0;
//  		while(key != 'q' )
//  			key = waitKey();
	
	cuApplyCropRandom(batchImg[batchId].m_devPoint, cropOutputs.m_devPoint, batch, outputDim);

	if(fabs(Config::instance()->getDistortion()) >= 0.1 || Config::instance()->getScale() >= 1 || Config::instance()->getRotation() >= 1)
		cuApplyDistortion(cropOutputs.m_devPoint, cropOutputs.m_devPoint, batch, outputDim);

	if (Config::instance()->getHorizontal()) {
		cuApplyHorizontal(cropOutputs.m_devPoint,
			cropOutputs.m_devPoint, batch, outputDim, RANDOM_HORIZONTAL);
	}

	if(Config::instance()->getWhiteNoise() >= 0.01){
		cuApplyWhiteNoise(cropOutputs.m_devPoint,
			cropOutputs.m_devPoint, batch, outputDim, Config::instance()->getWhiteNoise());
	}

	if (Config::instance()->getImageShow()) {
		for (int ff = batch - 1; ff >= 0; ff--) {
			showImg(batchImg[batchId][ff], 5);
			showImg(cropOutputs.m_vec[ff], 5);
			cv::waitKey(0);
		}
	}
}

void DataLayer::testData(int cropr, int cropc, 
	float scalex, float scaley,
	float rotate,
	int hori)
{
	cuApplyCrop(batchImg[batchId].m_devPoint,
		cropOutputs.m_devPoint, batch, outputDim,
		cropr, cropc);

	if(scalex || scaley || rotate){
		cuApplyScaleAndRotate(batch, outputDim, scalex, scaley, rotate);
		cuApplyDistortion(cropOutputs.m_devPoint, 
			cropOutputs.m_devPoint,
			batch, 
			outputDim);
	}

	if (hori == 1)
		cuApplyHorizontal(cropOutputs.m_devPoint,
		cropOutputs.m_devPoint, batch, outputDim, HORIZONTAL);

// 	if (Config::instance()->getImageShow() && scalex >= 0.0 && scaley >= 0 && scaley >= 0 && rotate >= 0) {
// 		for (int ff = batch - 1; ff >= 0; ff--) {
// 			showImg(batchImg[batchId][ff], 5);
// 			showImg(cropOutputs.m_vec[ff], 5);
// 			cv::waitKey(0);
// 		}
// 	}
}


void DataLayer::synchronize(){
	batchId = 1 - batchId;
	cudaStreamSynchronize(stream1);
}

void DataLayer::getBatchImageWithStreams(cuMatrixVector<float>& inputs, int start){
	int id = 1 - batchId;
	for(int i = 0; i < (int)batchImg[id].size(); i++){
		memcpy(batchImg[id][i]->getHost(), inputs[i + start]->getHost(), sizeof(float) * batchImg[id][i]->getLen());
		batchImg[id][i]->toGpu(stream1);
	}
}