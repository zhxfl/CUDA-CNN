#ifndef __CONV_COMBINE_FEATURE_MAP_CU_H__
#define __CONV_COMBINE_FEATURE_MAP_CU_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"


class ConvCFM: public LayerBase
{
public:
	void feedforward(){};
	void backpropagation(){};
	void getGrad(){};
	void updateWeight(){};
	ConvCFM(cuMatrix<double>* _inputs, int _size, int _skip,int _inputDim, int _amount, int _batch);

	~ConvCFM(){
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
	cuMatrix<double>* outputs;
	cuMatrix<double>* preDelta;
	cuMatrix<double>* curDelta; // size(curDelta) == size(outputs)
	int inputDim ;
	int outputDim;
	int amount;
	int batch;
	int NON_LINEARITY;
private:
	cuMatrixVector<double>* w;
	cuMatrixVector<double>* wgrad;
	cuMatrixVector<double>* b;
	cuMatrixVector<double>* bgrad;
};

#endif 