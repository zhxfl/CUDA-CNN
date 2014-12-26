#ifndef _READ_MNIST_DATA_H_
#define _READ_MNIST_DATA_H_

#include "cuMatrix.h"
#include <string>
#include <vector>
#include "util.h"
#include "cuMatrixVector.h"


/*read trainning data and lables*/
int readMnistData(cuMatrixVector<double> &x,
	cuMatrix<int>* &y, 
	std::string xpath,
	std::string ypath,
	int number_of_images,
	int flag);

#endif
