#ifndef __SOFT_MAX_H__
#define __SOFT_MAX_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"

class SoftMax: public LayerBase
{
public:
	void feedforward();
	void backpropagation();
	void getGrad();
	void updateWeight() ;
	void clearMomentum();
	void getCost(cuMatrix<double>*cost, int* y);

	cuMatrix<double>* getOutputs();
	cuMatrix<double>* getPreDelta();
	cuMatrix<double>* getCurDelta() ;

	void setPreDelta(cuMatrix<double>* _preDelta);

	void initRandom();
	void initFromCheckpoint(FILE* file);
	void save(FILE* file);


	SoftMax(std::string name);

private:
	cuMatrix<double>* inputs;
	cuMatrix<double>* outputs;
	cuMatrix<double>* curDelta;
	cuMatrix<double>* preDelta;

	cuMatrix<double>* w;
	cuMatrix<double>* wgrad;
	cuMatrix<double>* b;
	cuMatrix<double>* bgrad;
	cuMatrix<double>* groudTruth;

	cuMatrix<double>* momentum_w;
	cuMatrix<double>* momentum_b;

	int inputsize;
	int outputsize; 
	double lambda;
	int batch;
	int NON_LINEARITY;
};
#endif
