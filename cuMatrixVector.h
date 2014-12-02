#ifndef __CU_MATRIX_VECTOR_H_
#define __CU_MATRIX_VECTOR_H_

#include <vector>
#include "cuMatrix.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

template <class T>
class cuMatrixVector
{
public:
	cuMatrixVector(): m_hstPoint(0), m_devPoint(0){}
	~cuMatrixVector(){
		free(m_hstPoint);
		cudaFree(m_devPoint);
		m_vec.clear();
	}
	cuMatrix<T>* operator[](size_t index){
		if(index >= m_vec.size()){
			printf("cuMatrix Vector operator[] error\n");
			exit(0);
		}
		return m_vec[index];
	}
	size_t size(){
		return m_vec.size();
	}
	void push_back(cuMatrix<T>* m){
		m_vec.push_back(m);
	}
	vector<cuMatrix<T>*>m_vec;
	T** m_hstPoint;
	T** m_devPoint;
	void toGpu()
	{
		cudaError_t cudaStat;

		m_hstPoint = (T**)malloc(m_vec.size() * sizeof(T*));
		if(!m_hstPoint){
			printf("cuMatrixVector<T> malloc m_hstPoint fail\n");
			exit(0);
		}

		cudaStat = cudaMalloc((void**)&m_devPoint, sizeof(T*) * m_vec.size());
		if(cudaStat != cudaSuccess){
			printf("cuMatrixVector<T> cudaMalloc m_devPoint fail\n");
			exit(0);
		}

		for(int p = 0; p < m_vec.size(); p++){
			m_hstPoint[p] = m_vec[p]->devData;
		}

		cudaStat = cudaMemcpy(m_devPoint, m_hstPoint, sizeof(T*) * m_vec.size(), cudaMemcpyHostToDevice);
		if(cudaStat != cudaSuccess){
			printf("cuConvLayer::init cudaMemcpy w fail\n");
			exit(0);
		}
	}
};

#endif