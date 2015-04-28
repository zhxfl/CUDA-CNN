#ifndef __DATA_PRETREATMENT_CU_H__
#define __DATA_PRETREATMENT_CU_H__

#include "../common/cuMatrixVector.h"

void preProcessing(
	cuMatrixVector<float>&trainX,
	cuMatrixVector<float>&testX);

#endif
