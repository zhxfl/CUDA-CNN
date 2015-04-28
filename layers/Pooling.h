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
	void calCost(){};

	void initFromCheckpoint(FILE* file){};
	void save(FILE* file){};

	~Pooling(){
		delete pointX;
		delete pointY;
		delete outputs;
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

	void printParameter(){}

private:
	cuMatrix<float>* inputs;
	cuMatrix<int>   * pointX;
	cuMatrix<int>   * pointY;
	cuMatrix<float>* preDelta;
	cuMatrix<float>* outputs;
	cuMatrix<float>* curDelta; // size(curDelta) == size(outputs)
	int size;
	int skip;
	int batch;
	std::string type;/*MAX AVR*/
	int NON_LINEARITY;
};
#endif
