#ifndef __LAYERS_POOLING_H__
#define __LAYERS_POOLING_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"

//non-linearity
//#define NL_SIGMOID 0
//#define NL_TANH 1
//#define NL_RELU 2

class Pooling: public ConvLayerBase{
public:
	void feedforward();
	void backpropagation();
	void getGrad(){};
	void updateWeight(){};
	void clearMomentum(){};
	Pooling(std::string name);
	void getCost(cuMatrix<double>*cost, int* y = NULL);

	void initFromCheckpoint(FILE* file){};
	void save(FILE* file){};

	~Pooling(){
		delete pointX;
		delete pointY;
		delete outputs;
	}

	cuMatrix<double>* getOutputs(){
		return outputs;
	};

	cuMatrix<double>* getPreDelta(){
		return preDelta;
	}

	cuMatrix<double>* getCurDelta(){
		return curDelta;
	}

	void setPreDelta(cuMatrix<double>* _preDelta){
		preDelta = _preDelta;
	}

	int getOutputAmount(){
		return amount;
	}

	int getOutputDim(){
		return outputDim;
	}

	void printParameter(){}

private:
	cuMatrix<double>* inputs;
	cuMatrix<int>   * pointX;
	cuMatrix<int>   * pointY;
	cuMatrix<double>* preDelta;
	cuMatrix<double>* outputs;
	cuMatrix<double>* curDelta; // size(curDelta) == size(outputs)
	int size;
	int skip;
	int batch;
	int NON_LINEARITY;
};
#endif
