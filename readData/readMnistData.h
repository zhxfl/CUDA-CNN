#ifndef _READ_MNIST_DATA_H_
#define _READ_MNIST_DATA_H_

#include "../common/cuMatrix.h"
#include "../common/cuMatrixVector.h"
#include "../common/util.h"
#include "../common/MemoryMonitor.h"
#include <string>
#include <vector>


/*read trainning data and lables*/
int readMnistData(cuMatrixVector<double> &x,
	cuMatrix<int>* &y, 
	std::string xpath,
	std::string ypath,
	int number_of_images,
	int flag);

#endif
