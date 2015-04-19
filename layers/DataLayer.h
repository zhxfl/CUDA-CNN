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

	cuMatrix<double>* getOutputs(){return outputs;}
	cuMatrix<double>* getCurDelta(){return NULL;}

	int getOutputAmount(){
		return outputAmount;
	}

	int getOutputDim(){
		return outputDim;
	}

	void trainData();
	void testData(int cropr, int cropc, 
		double scalex, double scaley,
		double rotate,
		int hori);

	void printParameter(){};
	void synchronize();

	void getBatchImageWithStreams(cuMatrixVector<double>& inputs, int start);
private:
	cuMatrix<double>* outputs;
	cuMatrixVector<double>cropOutputs;
	cuMatrixVector<double>batchImg[2];/*batch size images*/
	int batchId;
	int batch;
	cudaStream_t stream1;
};
#endif
