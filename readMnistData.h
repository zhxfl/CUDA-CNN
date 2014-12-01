#ifndef _READ_DATA_H_
#define _READ_DATA_H_

#include "cuMatrix.h"
#include <string>
#include <vector>
#include "util.h"

/*read trainning data and lables*/
int readMnistData(std::vector<cuMatrix<double>*> &x,
	cuMatrix<double>* &y, 
	std::string xpath,
	std::string ypath,
	int number_of_images,
	int flag);

#endif