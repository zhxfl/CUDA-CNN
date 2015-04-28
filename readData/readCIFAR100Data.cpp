#include "readCIFAR10Data.h"
#include <string>
#include <fstream>
#include <vector>
#include <vector>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
using namespace std;


void read_batch(std::string filename, cuMatrixVector<float>&vec, cuMatrix<int>*&label, int number_of_images)
{
	ifstream file(filename.c_str(), ios::binary);
	if(file.is_open())
	{
		int n_rows = 32;
		int n_cols = 32;
		for(int i = 0; i < number_of_images; i++)
		{
			unsigned char type1 = 0;
			unsigned char type2 = 0;
			file.read((char*)& type1, sizeof(type1));
			file.read((char*)& type2, sizeof(type2));
			//printf("type1 = %d type2 = %d\n", type1, type2);
			cuMatrix<float>* channels = new cuMatrix<float>(n_rows, n_cols, 3);
			channels->freeCudaMem();
			int idx = vec.size();
			label->set(idx, 0, 0, type2);

			for(int ch = 0; ch < 3; ch++){
				for(int r = 0; r < n_rows; r++){
					for(int c = 0; c < n_cols; c++){
						unsigned char temp = 0;
						file.read((char*) &temp, sizeof(temp));
						channels->set(r, c , ch, float(temp) / 256.0f * 2.0f - 1.0f);
					}
				}
			}

			vec.push_back(channels);
		}
	}
}

void read_CIFAR100_Data(cuMatrixVector<float> &trainX,
	cuMatrixVector<float>&testX,
	cuMatrix<int>*&trainY,
	cuMatrix<int>*&testY)
{
	/*read the train data and train label*/
	string filename;
	filename = "cifar-100-binary/train.bin";
	
	trainY = new cuMatrix<int>(50000, 1, 1);
	read_batch(filename, trainX, trainY, 50000);
	
	//trainX.toGpu();
	trainY->toGpu();

	/*read the test data and test labels*/
	filename = "cifar-100-binary/test.bin";
	testY = new cuMatrix<int>(10000, 1, 1);
	read_batch(filename, testX, testY, 10000);
	//testX.toGpu();
	testY->toGpu();
}
