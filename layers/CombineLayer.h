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

	cuMatrix<double>* getOutputs() {return outputs;}
	cuMatrix<double>* getCurDelta(){return curDelta;}

	int getOutputAmount(){
		return outputAmount;
	}

	int getOutputDim(){
		return outputDim;
	}

	void printParameter(){};

private:
	cuMatrixVector<double> inputs;
	cuMatrixVector<double> preDelta;

	cuMatrix<double>*outputs;
	cuMatrix<double>*curDelta;

	cuMatrix<int>* inputsSkip;
	cuMatrix<int>* inputsCols;
	cuMatrix<int>* inputsChannels;

	int batch;
};
#endif
