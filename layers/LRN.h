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

	cuMatrix<float>* getOutputs(){
		return outputs;
	};

	cuMatrix<float>* getCurDelta(){
		return curDelta;
	}

	void setPreDelta(cuMatrix<float>* _preDelta){
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
	cuMatrix<float>* inputs;
	cuMatrix<float>* preDelta;
	cuMatrix<float>* outputs;
	cuMatrix<float>* curDelta; // size(curDelta) == size(outputs)
	
	float lrn_k;
	int    lrn_n;
	float lrn_alpha;
	float lrn_belta;

	int batch;
	int NON_LINEARITY;
};
#endif
