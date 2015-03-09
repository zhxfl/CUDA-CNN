#ifndef __FULL_CONNECT__H__
#define __FULL_CONNECT__H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"

class FullConnect: public LayerBase
{
public:
	void feedforward();
	void backpropagation();
	void getGrad();
	void updateWeight() ;
	void clearMomentum();
	void getCost(cuMatrix<double>*cost, int* y = NULL);
	void drop();
	void drop(double rate);

	cuMatrix<double>* getOutputs();
	cuMatrix<double>* getPreDelta();
	cuMatrix<double>* getCurDelta() ;

	void setPreDelta(cuMatrix<double>* _preDelta);

	void initRandom();
	void initFromCheckpoint(FILE* file);
	void save(FILE* file);


	FullConnect(std::string name);

private:
	cuMatrix<double>* inputs;
	cuMatrix<double>* inputs_format;//convert inputs(batch, size, channel) to (batch, size * channel)
	cuMatrix<double>* outputs;
	cuMatrix<double>* curDelta;
	cuMatrix<double>* preDelta;
	cuMatrix<double>* preDelta_format;

	cuMatrix<double>* dropW;
	cuMatrix<double>* afterDropW;

	cuMatrix<double>* w;
	cuMatrix<double>* wgrad;
	cuMatrix<double>* b;
	cuMatrix<double>* bgrad;

	cuMatrix<double>* momentum_w;
	cuMatrix<double>* momentum_b;

	int inputsize;
	int outputsize; 
	double lambda;
	int batch;
	double dropRate;
	int NON_LINEARITY;

private:
	void convert();
};
#endif
