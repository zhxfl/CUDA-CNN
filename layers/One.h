#ifndef __NET_IN_NET_CU_H__
#define __NET_IN_NET_CU_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"


class One: public ConvLayerBase
{
public:
	void feedforward();
	void backpropagation();
	void getGrad();
	void updateWeight();
	void clearMomentum();
	void calCost();
	
	One(std::string name);

	void initRandom();
	void initFromCheckpoint(FILE* file);
	void save(FILE* file);

	~One(){
		delete outputs;
	}

	cuMatrix<float>* getOutputs(){
		return outputs;
	};

	cuMatrix<float>* getCurDelta(){
		return curDelta;
	}

	int getOutputAmount(){
		return outputAmount;
	}

	int getOutputDim(){
		return outputDim;
	}

	virtual void printParameter(){
		char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		w->toCpu();
		sprintf(logStr, "weight:%f, %f:\n", w->get(0,0,0), w->get(0,1,0));
		LOG(logStr, "Result/log.txt");
		b->toCpu();
		sprintf(logStr, "bias  :%f, %f:\n", b->get(0,0,0), b->get(0,1,0));
		LOG(logStr, "Result/log.txt");
	}
private:
	cuMatrix<float>* inputs;
	cuMatrix<float>* preDelta;
	cuMatrix<float>* outputs;
	cuMatrix<float>* curDelta;
	float lambda;
private:
	cuMatrix<float>* w;
	cuMatrix<float>* wgrad;
	cuMatrix<float>* b;
	cuMatrix<float>* bgrad;
	cuMatrix<float>* momentum_w;
	cuMatrix<float>* momentum_b;
	cuMatrix<float>* wgradTmp;
};

#endif 
