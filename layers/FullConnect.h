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
	void calCost();
	void dropOut();
	//void drop(double rate);
	cuMatrix<double>* getOutputs();
	cuMatrix<double>* getCurDelta() ;

	void setPreDelta(cuMatrix<double>* _preDelta);

	void initRandom();
	void initFromCheckpoint(FILE* file);
	void save(FILE* file);
	virtual void printParameter(){
		char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		w->toCpu();
		sprintf(logStr, "weight:%lf, %lf;\n", w->get(0,0,0), w->get(0,1,0));
		LOG(logStr, "Result/log.txt");
		b->toCpu();
		sprintf(logStr, "bias  :%lf\n", b->get(0,0,0));
		LOG(logStr, "Result/log.txt");
	}

	FullConnect(std::string name);

private:
	cuMatrix<double>* inputs;
	cuMatrix<double>* inputs_format;//convert inputs(batch, size, channel) to (batch, size * channel)
	cuMatrix<double>* outputs;
	cuMatrix<double>* curDelta;
	cuMatrix<double>* preDelta;
	cuMatrix<double>* preDelta_format;

	//cuMatrix<double>* dropW;
	//cuMatrix<double>* afterDropW;
	cuMatrix<double>* drop;
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
