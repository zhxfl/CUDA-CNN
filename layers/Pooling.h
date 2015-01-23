#ifndef __LAYERS_POOLING_H__
#define __LAYERS_POOLING_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include "../common/util.h"

// non-linearity
//#define NL_SIGMOID 0
//#define NL_TANH 1
//#define NL_RELU 2

class Pooling: public LayerBase{
public:
	void feedforward();
	void backpropagation();
private:
	std::vector<cuMatrix<double>* >inputs;
	std::vector<cuMatrix<double>* >outputs;
	std::vector<cuMatrix<int>* >pointX;
	std::vector<cuMatrix<int>* >pointY;
	std::vector<cuMatrix<double>* >preDelta;
	cuMatrix<double>*curDelta; // size(curDelta) == size(outputs)
	int size;
	int skip;
	int NON_LINEARITY;
};
#endif
