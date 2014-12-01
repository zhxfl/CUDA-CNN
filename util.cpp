#include "util.h"
#include <opencv2/opencv.hpp>
using namespace cv;

void showImg(cuMatrix<double>* x, double scala)
{
	x->toCpu();
	double * data = (double*) malloc(sizeof(*data) * x->cols * x->rows);
	if(!data) {
		printf("util::showImg memory allocation failed");
		exit(0);
	}

	for(int i = 0; i < x->rows; i++)
	{
		for(int j = 0; j < x->cols; j++)
		{
			data[i * x->cols + j] = x->get(i,j);
		}
	}

	Mat src(x->rows, x->cols, CV_64FC1, data);
	
	Size size;
	size.width  = src.cols * scala;
	size.height = src.rows * scala;

	Mat dst(size.height, size.width, CV_64FC1);

	cv::resize(src, dst, size);

	static int id = 1;
	id++;
	char ch[10];
	sprintf(ch, "%d",id);
	namedWindow(ch, WINDOW_AUTOSIZE);
	imshow(ch, dst);
	free(data);
}

void DebugPrintf(cuMatrix<double>*x)
{
	x->toCpu();
	for(int i = 0; i < x->rows; i++)
	{
		for(int j = 0; j < x->cols; j++)
		{
			printf("%lf ", x->get(i,j));
		}printf("\n");
	}
}

void DebugPrintf(double* data, int len, int dim)
{
	for(int id = 0; id < len; id += dim*dim)
	{
		double* img = data + id;
		for(int i = 0; i < dim; i++)
		{
			for(int j = 0; j < dim; j++)
			{
				printf("%lf ", img[i * dim + j]);
			}printf("\n");
		}
	}
}

void LOG(char* str, char* file)
{
	FILE* f = fopen(file,"a");
	fprintf(f,"%s\n",str);
	fclose(f);
}