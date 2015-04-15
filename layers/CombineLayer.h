/*
ref : ImageNet Classification with Deep Convolutional Neural Networks
*/
#ifndef __LAYERS_LRN_H__
#define __LAYERS_LRN_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"

//non-linearity
//#define NL_SIGMOID 0
//#define NL_TANH 1
//#define NL_RELU 2

class LRN: public ConvLayerBase{
public:
	LRN(std::string name);

	void feedforward();
	void backpropagation();
	void getGrad(){};
	void updateWeight(){};
	void clearMomentum(){};

	void calCost(){};
	void initFromCheckpoint(FILE* file){};
	void save(FILE* file){};

	~LRN(){
		delete outputs;
		delete curDelta;
	}

	cuMatrix<double>* getOutputs(){
		return outputs;
	};

	cuMatrix<double>* getCurDelta(){
		return curDelta;
	}

	void setPreDelta(cuMatrix<double>* _preDelta){
		preDelta = _preDelta;
	}

	int getOutputAmount(){
		return outputAmount;
	}

	int getOutputDim(){
		return outputDim;
	}

	void printParameter(){};

private:
	cuMatrix<double>* inputs;
	cuMatrix<double>* preDelta;
	cuMatrix<double>* outputs;
	cuMatrix<double>* curDelta; // size(curDelta) == size(outputs)
	
	double lrn_k;
	int    lrn_n;
	double lrn_alpha;
	double lrn_belta;

	int batch;
	int NON_LINEARITY;
};
#endif
