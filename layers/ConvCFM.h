#ifndef __CONV_COMBINE_FEATURE_MAP_CU_H__
#define __CONV_COMBINE_FEATURE_MAP_CU_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"


class ConvCFM: public ConvLayerBase
{
public:
	void feedforward();
	void backpropagation();
	void getGrad();
	void updateWeight();
	void clearMomentum();
	void getCost(cuMatrix<double>*cost, int* y = NULL);
	
	ConvCFM(std::string name);


	void initRandom();
	void initFromCheckpoint(FILE* file);
	void save(FILE* file);

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

	int getOutputAmount(){
		return outputAmount;
	}

	int getInputDim(){
		return inputDim;
	}

	int getOutputDim(){
		return outputDim;
	}

private:
	cuMatrixVector<double>* inputs_1;
	cuMatrix<double>* inputs_2;
	cuMatrix<double>* preDelta;
	cuMatrix<double>* outputs;
	cuMatrix<double>* curDelta; // size(curDelta) == size(outputs)
	//int outputAmount;  outputAmount = inputAmount * amount
	int kernelSize;
	int padding;
	int batch;
	int NON_LINEARITY;
	int cfm;
	double lambda;
private:
	cuMatrixVector<double> w;
	cuMatrixVector<double> wgrad;
	cuMatrixVector<double> b;
	cuMatrixVector<double> bgrad;
	cuMatrixVector<double> momentum_w;
	cuMatrixVector<double> momentum_b;
	cuMatrix<double>* wgradTmp;
};

#endif 
