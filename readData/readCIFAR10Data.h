#ifndef _READ_CIFAR10_DATA_H_
#define _READ_CIFAR10_DATA_H_

#include "../cuMatrix.h"
#include "../cuMatrixVector.h"
#include "../util.h"
#include "../MemoryMonitor.h"
#include <string>
#include <vector>


/*read trainning data and lables*/
void read_CIFAR10_Data(cuMatrixVector<double> &trainX,
	cuMatrixVector<double>&testX,
	cuMatrix<int>*&trainY,
	cuMatrix<int>*&testY);


#endif
