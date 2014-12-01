#ifndef _UTIL_H_
#define _UTIL_H_

#include "cuMatrix.h"

#define MAX_THREADS 1024

// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2

void showImg(cuMatrix<double>* x, double scala);
void DebugPrintf(cuMatrix<double>*x);
void DebugPrintf(double* data, int len, int dim);
void LOG(char* str, char* file);

#endif