#ifndef __LAYERS_POOLING_H__
#define __LAYERS_POOLING_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"

// non-linearity
//#define NL_SIGMOID 0
//#define NL_TANH 1
//#define NL_RELU 2

class Pooling: public LayerBase{
public:
	void feedforward();
	void backpropagation();
	void getGrad(){};
	void updateWeight(){};
	void clearMomentum(){};
	Pooling(cuMatrix<double>* _inputs, int _size, int _skip,int _inputDim, int _amount, int _batch);

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

private:
	cuMatrix<double>* inputs;
	cuMatrix<int>   * pointX;
	cuMatrix<int>   * pointY;
	cuMatrix<double>* preDelta;
	cuMatrix<double>* outputs;
	cuMatrix<double>* curDelta; // size(curDelta) == size(outputs)
	int size;
	int skip;
	int inputDim ;
	int outputDim;
	int amount;
	int batch;
	int NON_LINEARITY;
};
#endif
