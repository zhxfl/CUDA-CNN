#ifndef _READ_CHINESE_DATA_H_
#define _READ_CHINESE_DATA_H_

#include "../cuMatrix.h"
#include "../cuMatrixVector.h"
#include "../util.h"
#include "../MemoryMonitor.h"
#include <string>
#include <vector>

void readChineseData(cuMatrixVector<double> &trainX,
	cuMatrixVector<double>&testX,
	cuMatrix<int>*&trainY,
	cuMatrix<int>*&testY);


#endif
