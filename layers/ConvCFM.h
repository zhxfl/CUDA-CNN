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
	void feedforward();
	void backpropagation();
	void getGrad();
	void updateWeight();
	void clearMomemtum();

	ConvCFM(cuMatrix<double>* _inputs, 
		int _inputAmount,
		int _amount,
		int _kernelSize,
		int _padding,
		int _inputDim,
		int _batch, 
		double _weight_decay,
		double _lambda,
		int cfm,
		int _NON_LINEARITY);

	ConvCFM(cuMatrixVector<double>* _inputs, 
		int _inputAmount,
		int _amount,
		int _kernelSize,
		int _padding,
		int _inputDim,
		int _batch, 
		double _weight_decay,
		double _lambda,
		int cfm,
		int _NON_LINEARITY);

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

	void setPreDelta(cuMatrix<double>* _preDelta){
		preDelta = _preDelta;
	}

	int getOutputAmount(){
		return outputAmount;
	}

private:
	cuMatrixVector<double>* inputs_1;
	cuMatrix<double>* inputs_2;
	cuMatrix<double>* preDelta;
	cuMatrix<double>* outputs;
	cuMatrix<double>* curDelta; // size(curDelta) == size(outputs)
	int inputDim ;
	int outputDim;
	int inputAmount;
	int amount;
	int outputAmount; // outputAmount = inputAmount * amount
	int kernelSize;
	int padding;
	int batch;
	int NON_LINEARITY;
	int cfm;
	double weight_decay;
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