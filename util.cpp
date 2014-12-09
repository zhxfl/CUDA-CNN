#include "util.h"
#include <opencv2/opencv.hpp>
#include "Config.h"
using namespace cv;

int getCV_64()
{
	int cv_64;
	if(Config::instance()->getChannels() == 1){
		cv_64 = CV_64FC1;
	}
	else if(Config::instance()->getChannels() == 3){
		cv_64 = CV_64FC3;
	}
	else if(Config::instance()->getChannels() == 4){
		cv_64 = CV_64FC4;
	}
	return cv_64;
}

void showImg(cuMatrix<double>* x, double scala)
{
	x->toCpu();

	int CV_64;
	if(x->channels == 1){
		CV_64 = CV_64FC1;
	}
	else if(x->channels == 3){
		CV_64 = CV_64FC3;
	}
	else if(x->channels == 4){
		CV_64 = CV_64FC4;
	}
	Mat src(x->rows, x->cols, CV_64);;


	for(int i = 0; i < x->rows; i++)
	{
		for(int j = 0; j < x->cols; j++)
		{
			if(x->channels == 1){
				src.at<double>(i, j) = x->get(i, j, 0);
			}
			else if(x->channels == 3){
				src.at<Vec3d>(i, j) = 
					Vec3d(
					x->get(i, j, 0),
					x->get(i, j, 1), 
					x->get(i, j, 2));
			}else if(x->channels == 4){
				src.at<Vec4d>(i, j) = 
					Vec4d(
					x->get(i, j, 0),
					x->get(i, j, 1),
					x->get(i, j, 2),
					x->get(i, j, 3));
			}
		}
	}

	Size size;
	size.width  = src.cols * scala;
	size.height = src.rows * scala;

	Mat dst(size.height, size.width, CV_64);

	cv::resize(src, dst, size);

	static int id = 0;
	id++;
	char ch[10];
	sprintf(ch, "%d", id);
	namedWindow(ch, WINDOW_AUTOSIZE);
	imshow(ch, dst);
}

void DebugPrintf(cuMatrix<double>*x)
{
	x->toCpu();
	for(int c = 0; c < x->channels; c++)
	{
		for(int i = 0; i < x->rows; i++)
		{
			for(int j = 0; j < x->cols; j++)
			{
				printf("%lf ", x->get(i, j, c));
			}printf("\n");
		}
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