#include "util.h"
#include <opencv2/opencv.hpp>
#include "Config.h"
#include <time.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

int getSharedMemory(vector<unsigned int>& vec) {
    int dev_num = 0;
    if(!vec.empty())vec.clear();

    cudaError_t error_id = cudaGetDeviceCount(&dev_num);
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int) error_id,
                cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }
    for (int dev = 0; dev < dev_num; dev++) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        vec.push_back((unsigned int)deviceProp.sharedMemPerBlock);
    }
    return dev_num;
}

bool checkSharedMemory(int id, size_t MemorySize){
    static std::vector<unsigned int>ret;
    if(ret.size() == 0)
    {
        getSharedMemory(ret);
    }
    if(ret.size() > (size_t)id){
        if(ret[id] >= MemorySize){
            return false;
            return true;
        }
        else{
            return false;
            LOG("getSharedMemory error", "result/log.txt");
            exit(0);
        }
    }else{
        return false;
        LOG("getSharedMemory error", "result/log.txt");
        exit(0);
    }
}

int getCV_32()
{
    int cv_32;
    if(Config::instance()->getChannels() == 1){
        cv_32 = CV_32FC1;
    }
    else if(Config::instance()->getChannels() == 3){
        cv_32 = CV_32FC3;
    }
    else if(Config::instance()->getChannels() == 4){
        cv_32 = CV_32FC4;
    }
    return cv_32;
}

void showImg(cuMatrix<float>* x, float scala)
{
    x->toCpu();

    int CV_32;
    if(x->channels == 1){
        CV_32 = CV_32FC1;
    }
    else if(x->channels == 3){
        CV_32 = CV_32FC3;
    }
    else if(x->channels == 4){
        CV_32 = CV_32FC4;
    }
    Mat src(x->rows, x->cols, CV_32);;


    for(int i = 0; i < x->rows; i++)
    {
        for(int j = 0; j < x->cols; j++)
        {
            if(x->channels == 1){
                src.at<float>(i, j) = x->get(i, j, 0);
            }
            else if(x->channels == 3){
                src.at<Vec3f>(i, j) = 
                    Vec3f(
                            x->get(i, j, 0),
                            x->get(i, j, 1), 
                            x->get(i, j, 2));
            }else if(x->channels == 4){
                src.at<Vec4f>(i, j) = 
                    Vec4f(
                            x->get(i, j, 0),
                            x->get(i, j, 1),
                            x->get(i, j, 2),
                            x->get(i, j, 3));
            }
        }
    }

    Size size;
    size.width  = int(1.0f * src.cols * scala);
    size.height = int(1.0f * src.rows * scala);


    Mat dst(size.height, size.width, CV_32);

    cv::resize(src, dst, size);

    static int id = 0;
    id++;
    char ch[10];
    sprintf(ch, "%d", id);
    namedWindow(ch, WINDOW_AUTOSIZE);
    cv::imshow(ch, dst);
}

void DebugPrintf(cuMatrix<float>*x)
{
    FILE *file = fopen("DEBUG.txt", "w+");
    x->toCpu();
    for(int c = 0; c < x->channels; c++)
    {
        for(int i = 0; i < x->rows; i++)
        {
            for(int j = 0; j < x->cols; j++)
            {
                fprintf(file, "%f ", x->get(i, j, c));
            }fprintf(file, "\n");
        }
    }
}

void DebugPrintf(float* data, int len, int dim)
{
    for(int id = 0; id < len; id += dim*dim)
    {
        float* img = data + id;
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                printf("%f ", img[i * dim + j]);
            }printf("\n");
        }
    }
}

void LOG(const char* str, const char* file)
{
    FILE* f = fopen(file,"a");
    printf("%s", str);
    fprintf(f,"%s",str);
    fclose(f);
}


void createGaussian(float* gaussian, float dElasticSigma1, float dElasticSigma2,
        int rows, int cols, int channels, float epsilon)
{
    int iiMidr = rows >> 1;
    int iiMidc = cols >> 1;

    float _sum = 0.0;
    for(int row = 0; row < rows; row++)
    {
        for(int col = 0; col < cols; col++)
        {
            float val1 = 1.0f / (dElasticSigma1 * dElasticSigma2 * 2.0f * 3.1415926535897932384626433832795f);
            float val2 = 1.0f * (row-iiMidr)*(row-iiMidr) / (dElasticSigma1 * dElasticSigma1) + 1.0f * (col-iiMidc)*(col-iiMidc) / (dElasticSigma2 * dElasticSigma2) 
                + 2.0f * (row - iiMidr) * (col - iiMidc) / (dElasticSigma1 * dElasticSigma2);
            gaussian[row * cols + col] = val1 * exp(-1.0f * val2);
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
            float val = gaussian[row * cols + col] / _sum;
            //val = val * 2.0 - 0.5;
            //val = val * epsilon;
            gaussian[row * cols + col] = val * epsilon;
            //printf("%f ", val * epsilon);
        }//printf("\n");
    }
    //printf("\n\n");
}


void dropDelta(cuMatrix<float>* M, float cuDropProb)
{
    //srand(clock());
    for(int c = 0; c < M->channels; c++){
        //cv::Mat ran = cv::Mat::zeros(M->rows, M->cols, CV_64FC1);
        //cv::theRNG().state = clock();
        //randu(ran, cv::Scalar(0), cv::Scalar(1.0));
        for(int i = 0; i < M->rows; i++){
            for(int j = 0; j < M->cols; j++){
                float r = 1.0f * rand() / RAND_MAX;
                if(r < cuDropProb)
                    M->set(i, j, c, 0.0);
                else 
                    M->set(i, j, c, 1.0);
            }
        }
    }
    M->toGpu();
}


void dropScale(cuMatrix<float>* M, float cuDropProb)
{
    for(int c = 0; c < M->channels; c++){
        for(int i = 0; i < M->rows; i++){
            for(int j = 0; j < M->cols; j++){
                M->set(i, j, c, 1.0 - cuDropProb);
            }
        }
    }
    M->toGpu();
}


void initMatrix(cuMatrix<float>* M, float initW)
{
    for(int c = 0; c < M->channels; c++){
        srand(clock());
        Mat matrix2xN = Mat::zeros(M->rows,M->cols,CV_64FC1);
        randn(matrix2xN, 0, initW); 
        for(int i = 0; i < matrix2xN.rows; i++){
            for(int j = 0; j < matrix2xN.cols; j++){
                M->set(i,j,c, matrix2xN.at<float>(i, j));
                printf("%f ", matrix2xN.at<float>(i, j));
            }printf("\n");
        }
        printf("\n\n");
    }

    M->toGpu();
}

void checkMatrixIsSame(cuMatrix<float>*x, cuMatrix<float>*y)
{
    Assert(x->rows == y->rows);
    Assert(x->cols == y->cols);
    Assert(x->channels == y->channels);
    for(int i = 0; i < x->rows; i++){
        for(int j = 0; j < x->cols; j++){
            for(int k = 0; k < x->channels; k++){
                float t = x->get(i, j, k) - y->get(i, j, k);
                if(fabs(t) > 0.001){
                    printf("\n%d %d %d %f %f %f\n", i, j, k, x->get(i,j, k), y->get(i,j,k), t);
                }
                Assert(fabs(t) < 0.001);
            }
        }
    }
}
