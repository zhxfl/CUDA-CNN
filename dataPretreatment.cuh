#ifndef __DATA_PRETREATMENT_CU_H__
#define __DATA_PRETREATMENT_CU_H__

#include "cuMatrixVector.h"

void preProcessing(
	cuMatrixVector<double>&trainX,
	cuMatrixVector<double>&testX);

#endif
