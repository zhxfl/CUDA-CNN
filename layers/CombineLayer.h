/*
ref : ImageNet Classification with Deep Convolutional Neural Networks
*/
#ifndef __LAYERS_COMBINE_LAYER_H__
#define __LAYERS_COMBINE_LAYER_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include <map>
#include "../common/util.h"


class CombineLayer: public ConvLayerBase{
public:
	CombineLayer(std::string name);

	void feedforward();
	void backpropagation();
	void getGrad(){};
	void updateWeight(){};
	void clearMomentum(){};

	void calCost(){};
	void initFromCheckpoint(FILE* file){};
	void save(FILE* file){};

	~CombineLayer(){
		delete outputs;
		delete curDelta;
	}

	cuMatrix<float>* getOutputs() {return outputs;}
	cuMatrix<float>* getCurDelta(){return curDelta;}

	int getOutputAmount(){
		return outputAmount;
	}

	int getOutputDim(){
		return outputDim;
	}

	void printParameter(){};

private:
	cuMatrixVector<float> inputs;
	cuMatrixVector<float> preDelta;

	cuMatrix<float>*outputs;
	cuMatrix<float>*curDelta;

	cuMatrix<int>* inputsSkip;
	cuMatrix<int>* inputsCols;
	cuMatrix<int>* inputsChannels;

	int batch;
};
#endif
