/*
ref : ImageNet Classification with Deep Convolutional Neural Networks
*/
#ifndef __LAYERS_DATA_LAYER_H__
#define __LAYERS_DATA_LAYER_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include <map>
//#include <thread>
#include "../common/util.h"


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
	cuMatrix<float>* outputs;
	cuMatrixVector<float>cropOutputs;
	cuMatrixVector<float>batchImg[2];/*batch size images*/
	int batchId;
	int batch;
	cudaStream_t stream1;
    //std::vector<std::thread>threads;
};
#endif
