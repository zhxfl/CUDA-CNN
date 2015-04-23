#include "util.h"
#include <opencv2/opencv.hpp>
#include "Config.h"
#include <time.h>

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
	cv::imshow(ch, dst);
}

void DebugPrintf(cuMatrix<double>*x)
{
	FILE *file = fopen("DEBUG.txt", "w+");
	x->toCpu();
	for(int c = 0; c < x->channels; c++)
	{
		for(int i = 0; i < x->rows; i++)
		{
			for(int j = 0; j < x->cols; j++)
			{
				fprintf(file, "%lf ", x->get(i, j, c));
			}fprintf(file, "\n");
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
	printf("%s", str);
	fprintf(f,"%s",str);
	fclose(f);
}


void createGaussian(double* gaussian, double dElasticSigma1, double dElasticSigma2,
	int rows, int cols, int channels, double epsilon)
{
	int iiMidr = rows >> 1;
	int iiMidc = cols >> 1;

	double _sum = 0.0;
	for(int row = 0; row < rows; row++)
	{
		for(int col = 0; col < cols; col++)
		{
			double val1 = 1.0 / (dElasticSigma1 * dElasticSigma2 * 2.0 * 3.1415926535897932384626433832795);
			double val2 = (row-iiMidr)*(row-iiMidr) / (dElasticSigma1 * dElasticSigma1) + (col-iiMidc)*(col-iiMidc) / (dElasticSigma2 * dElasticSigma2) 
				+ 2.0 * (row - iiMidr) * (col - iiMidc) / (dElasticSigma1 * dElasticSigma2);
			gaussian[row * cols + col] = val1 * exp(-1.0 * val2);
			//gaussian[row * cols + col] = exp(gaussian[row * cols + col]);
			_sum += gaussian[row * cols + col];
// 			if(_max < fabs(gaussian[row * cols + col]))
// 			{
// 				_max = fabs(gaussian[row * cols + col]);
// 			}
		}
	}
	for(int row = 0; row < rows; row++)
	{
		for(int col = 0; col < cols; col++)
		{
			double val = gaussian[row * cols + col] / _sum;
			//val = val * 2.0 - 0.5;
			//val = val * epsilon;
			gaussian[row * cols + col] = val * epsilon;
			//printf("%lf ", val * epsilon);
		}//printf("\n");
	}
	//printf("\n\n");
}


void dropDelta(cuMatrix<double>* M, double cuDropProb)
{
	srand(clock());
	for(int c = 0; c < M->channels; c++){
		//cv::Mat ran = cv::Mat::zeros(M->rows, M->cols, CV_64FC1);
		//cv::theRNG().state = clock();
		//randu(ran, cv::Scalar(0), cv::Scalar(1.0));
		for(int i = 0; i < M->rows; i++){
			for(int j = 0; j < M->cols; j++){
				double r = 1.0 * rand() / RAND_MAX;
				if(r < cuDropProb)
					M->set(i, j, c, 0.0);
				else 
					M->set(i, j, c, 1.0);
			}
		}
	}
	M->toGpu();
}


void initMatrix(cuMatrix<double>* M, double initW)
{
	for(int c = 0; c < M->channels; c++){
		srand(time(NULL));
		Mat matrix2xN = Mat::zeros(M->rows,M->cols,CV_64FC1);
		randn(matrix2xN, 0, initW); 
		for(int i = 0; i < matrix2xN.rows; i++){
			for(int j = 0; j < matrix2xN.cols; j++){
				M->set(i,j,c, matrix2xN.at<double>(i, j));
				printf("%lf ", matrix2xN.at<double>(i, j));
			}printf("\n");
		}
		printf("\n\n");
	}

	M->toGpu();
}