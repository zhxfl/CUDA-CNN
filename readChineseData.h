#ifndef _READ_CHINESE_DATA_H_
#define _READ_CHINESE_DATA_H_

#include "cuMatrix.h"
#include <string>
#include <vector>
#include "util.h"
#include "cuMatrixVector.h"

void readChineseData(cuMatrixVector<double> &trainX,
	cuMatrixVector<double>&testX,
	cuMatrix<int>*&trainY,
	cuMatrix<int>*&testY);

#endif
