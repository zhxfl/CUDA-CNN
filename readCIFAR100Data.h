#ifndef _READ_CIFAR100_DATA_H_
#define _READ_CIFAR100_DATA_H_

#include "cuMatrix.h"
#include <string>
#include <vector>
#include "util.h"
#include "cuMatrixVector.h"

/*read trainning data and lables*/
void read_CIFAR100_Data(cuMatrixVector<double> &trainX,
	cuMatrixVector<double>&testX,
	cuMatrix<int>*&trainY,
	cuMatrix<int>*&testY);

#endif
