#ifndef __NET_IN_NET_CU_H__
#define __NET_IN_NET_CU_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"


class NIN: public ConvLayerBase
{
public:
	void feedforward();
	void backpropagation();
	void getGrad();
	void updateWeight();
	void clearMomentum();
	void calCost();
	
	NIN(std::string name);

	void initRandom();
	void initFromCheckpoint(FILE* file);
	void save(FILE* file);

	~NIN(){
		delete outputs;
	}

	cuMatrix<double>* getOutputs(){
		return outputs;
	};

	cuMatrix<double>* getCurDelta(){
		return curDelta;
	}

	int getOutputAmount(){
		return outputAmount;
	}

	int getOutputDim(){
		return outputDim;
	}

	virtual void printParameter(){
		printf("%s:\n",m_name.c_str());
		w->toCpu();
		printf("weight:%lf, %lf:\n", w->get(0,0,0), w->get(0,1,0));
		b->toCpu();
		printf("bias  :%lf, %lf:\n", b->get(0,0,0), b->get(0,1,0));
	}
private:
	cuMatrix<double>* inputs;
	cuMatrix<double>* preDelta;
	cuMatrix<double>* outputs;
	cuMatrix<double>* curDelta;
	double lambda;
private:
	cuMatrix<double>* w;
	cuMatrix<double>* wgrad;
	cuMatrix<double>* b;
	cuMatrix<double>* bgrad;
	cuMatrix<double>* momentum_w;
	cuMatrix<double>* momentum_b;
	cuMatrix<double>* wgradTmp;
};

#endif 
