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
int  getCV_64();
void createGaussian(double* gaussian, double dElasticSigma1, double dElasticSigma2,
	int rows, int cols, int channels, double epsilon);

#define  cuAssert( X ) if ( !(X) ) {printf("tid %d: FILE=%s, LINE=%d\n", threadIdx.x, __FILE__, __LINE__); return; }
#define  Assert( X ) if ( !(X) ) {printf("FILE=%s, LINE=%d\n", __FILE__, __LINE__); return; }
#endif
