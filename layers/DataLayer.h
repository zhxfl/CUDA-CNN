/*
ref : ImageNet Classification with Deep Convolutional Neural Networks
*/
#ifndef __LAYERS_DATA_LAYER_H__
#define __LAYERS_DATA_LAYER_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include <map>
#include "../common/util.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui.hpp"


class DataLayer: public ConvLayerBase{
public:
	DataLayer(std::string name);

	void feedforward(); /*distortion the data*/
	void backpropagation(){};
	void getGrad(){};
	void updateWeight(){};
	void clearMomentum(){};

	void calCost(){};
	void initFromCheckpoint(FILE* file){};
	void save(FILE* file){};

	~DataLayer(){
		delete outputs;
		checkCudaErrors(cudaStreamDestroy(stream1));
	}

	cuMatrix<float>* getOutputs(){return outputs;}
	cuMatrix<float>* getCurDelta(){return NULL;}

	int getOutputAmount(){
		return outputAmount;
	}

	int getOutputDim(){
		return outputDim;
	}

	void trainData();

	void testData(int cropr, int cropc, 
		float scalex, float scaley,
		float rotate,
		int hori);

	void printParameter(){};
	void synchronize();
	
	void getBatchImageWithStreams(cuMatrixVector<float>& inputs, int start);
private:
	void readEigen(cv::Mat& eigenValues, cv::Mat& eigenVecotrs);
	cuMatrix<float>* outputs;
	cuMatrixVector<float>cropOutputs;
	cuMatrixVector<float>batchImg[2];/*batch size images*/
	cuMatrix<float>* color_noise;
	int batchId;
	int batch;
	cudaStream_t stream1;
	cv::Mat eigenValues;
	cv::Mat eigenVectors;
};
#endif
