#ifndef __LAYERS_BASE_CU_H__
#define __LAYERS_BASE_CU_H__

#include "../common/cuMatrix.h"

class LayerBase
{
public:
	virtual void feedforward() = 0;
	virtual void backpropagation() = 0;
	virtual void getGrad() = 0;
	virtual void updateWeight() = 0;
	virtual void clearMomentum() = 0;

	virtual cuMatrix<double>* getOutputs() = 0;
	virtual cuMatrix<double>* getPreDelta() = 0;
	virtual cuMatrix<double>* getCurDelta() = 0;

	virtual void setPreDelta(cuMatrix<double>* _preDelta) = 0;
};

#endif
