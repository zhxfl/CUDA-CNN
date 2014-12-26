#include "cuMatrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"


/*matrix multiply*/
/*z = x * y*/
void matrixMul(const cuMatrix<double>* x, const cuMatrix<double>*y, cuMatrix<double>*z, cublasHandle_t handle)
{
	cublasStatus_t stat;
	double alpha = 1.0;
	double beta = 0.0;
	stat = cublasDgemm(
		handle, 
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		y->cols,
		x->rows,
		y->rows,
		&alpha,
		y->devData,
		y->cols,
		x->devData,
		x->cols,
		&beta,
		z->devData,
		z->cols);
	cudaDeviceSynchronize();
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf("matrixMul cublasSgemm error\n");
		cudaFree(x->devData);
		cudaFree(y->devData);
		cudaFree(z->devData);
		exit(0);
	}
}

/*z = T(x) * y*/
void matrixMulTA(const cuMatrix<double>* x, const cuMatrix<double>*y, cuMatrix<double>*z, cublasHandle_t handle)
{
	cublasStatus_t stat;
	double alpha = 1.0;
	double beta = 0.0;
	stat = cublasDgemm(
		handle, 
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		y->cols,
		x->cols,
		y->rows,
		&alpha,
		y->devData,
		y->cols,
		x->devData,
		x->cols,
		&beta,
		z->devData,
		z->cols);
	cudaDeviceSynchronize();
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf("matrixMulTA cublasSgemm error\n");
		cudaFree(x->devData);
		cudaFree(y->devData);
		cudaFree(z->devData);
		exit(0);
	}
}

/*z = x * T(y)*/
void matrixMulTB(const cuMatrix<double>* x, const cuMatrix<double>*y, cuMatrix<double>*z, cublasHandle_t handle)
{
	cublasStatus_t stat;
	double alpha = 1.0;
	double beta = 0.0;
	stat = cublasDgemm(
		handle, 
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		y->rows,
		x->rows,
		y->cols,
		&alpha,
		y->devData,
		y->cols,
		x->devData,
		x->cols,
		&beta,
		z->devData,
		z->cols);
	cudaDeviceSynchronize();
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf("matrixMulTB cublasSgemm error\n");
		cudaFree(x->devData);
		cudaFree(y->devData);
		cudaFree(z->devData);
		exit(0);
	}
}
