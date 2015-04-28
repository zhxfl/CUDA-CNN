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

	cuMatrix<float>* getOutputs();
	cuMatrix<float>* getCurDelta() ;

	void setPreDelta(cuMatrix<float>* _preDelta);

	void initRandom();
	void initFromCheckpoint(FILE* file);
	void save(FILE* file);
	virtual void printParameter(){
		char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		w->toCpu();
		sprintf(logStr, "weight:%f, %f;\n", w->get(0,0,0), w->get(0,1,0));
		LOG(logStr, "Result/log.txt");
		b->toCpu();
		sprintf(logStr, "bias  :%f\n", b->get(0,0,0));
		LOG(logStr, "Result/log.txt");
	}

	FullConnect(std::string name);

private:
	cuMatrix<float>* inputs;
	cuMatrix<float>* inputs_format;//convert inputs(batch, size, channel) to (batch, size * channel)
	cuMatrix<float>* outputs;
	cuMatrix<float>* curDelta;
	cuMatrix<float>* preDelta;
	cuMatrix<float>* preDelta_format;

	//cuMatrix<float>* dropW;
	//cuMatrix<float>* afterDropW;
	cuMatrix<float>* drop;
	cuMatrix<float>* w;
	cuMatrix<float>* wgrad;
	cuMatrix<float>* b;
	cuMatrix<float>* bgrad;

	cuMatrix<float>* momentum_w;
	cuMatrix<float>* momentum_b;

	int inputsize;
	int outputsize; 
	float lambda;
	int batch;
	float dropRate;
	int NON_LINEARITY;

private:
	void convert();
};
#endif
