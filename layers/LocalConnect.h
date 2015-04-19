#ifndef __LOCAL_CONNECT_CU_H__
#define __LOCAL_CONNECT_CU_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"
#include "../common/cuMatrixVector.h"


class LocalConnect: public ConvLayerBase
{
public:
	void feedforward();
	void backpropagation();
	void getGrad();
	void updateWeight();
	void clearMomentum();
	void calCost();
	
	LocalConnect(std::string name);

	void initRandom();
	void initFromCheckpoint(FILE* file);
	void save(FILE* file);

	~LocalConnect(){
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
		char logStr[1024];
		sprintf(logStr, "%s:\n",m_name.c_str());
		LOG(logStr, "Result/log.txt");
		w[0]->toCpu();
		sprintf(logStr, "weight:%lf, %lf;\n", w[0]->get(0,0,0), w[0]->get(0,1,0));
		LOG(logStr, "Result/log.txt");
		b[0]->toCpu();
		sprintf(logStr, "bias  :%lf\n", b[0]->get(0,0,0));
		LOG(logStr, "Result/log.txt");
	}
private:
	cuMatrix<double>* inputs;
	cuMatrix<double>* preDelta;
	cuMatrix<double>* outputs;
	cuMatrix<double>* curDelta;
	int kernelSize;
	int batch;
	int NON_LINEARITY;
	double lambda;
	int localKernelSize;
private:
	cuMatrixVector<double> w;
	cuMatrixVector<double> wgrad;
	cuMatrixVector<double> b;
	cuMatrixVector<double> bgrad;
	cuMatrixVector<double> momentum_w;
	cuMatrixVector<double> momentum_b;
	cuMatrixVector<double> wgradTmp;
};

#endif 
