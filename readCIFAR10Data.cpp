#include "cuMatrix.h"
#include "readCIFAR10Data.h"
#include <string>
#include <fstream>
#include <vector>
#include "util.h"
#include <vector>
#include "cuMatrixVector.h"
#include <opencv2/opencv.hpp>
#include <stdlib.h>
using namespace std;

void read_batch(std::string filename, cuMatrixVector<double>&vec, cuMatrix<int>*&label)
{
	ifstream file(filename.c_str(), ios::binary);
	if(file.is_open())
	{
		int number_of_imgages = 10000;
		int n_rows = 32;
		int n_cols = 32;
		for(int i = 0; i < number_of_imgages; i++)
		{
			unsigned char tplabel = 0;
			file.read((char*)& tplabel, sizeof(tplabel));
			cuMatrix<double>* channels = new cuMatrix<double>(n_rows, n_cols, 3);
			int idx = vec.size();
			label->set(idx, 0, 0, tplabel);

			for(int ch = 0; ch < 3; ch++){
				for(int r = 0; r < n_rows; r++){
					for(int c = 0; c < n_cols; c++){
						unsigned char temp = 0;
						file.read((char*) &temp, sizeof(temp));
						channels->set(r, c , ch, double(temp) / 256.0 * 2.0 - 1.0);
					}
				}
			}
			channels->toGpu();
			vec.push_back(channels);
		}
	}

}

void read_CIFAR10_Data(cuMatrixVector<double> &trainX,
	cuMatrixVector<double>&testX,
	cuMatrix<int>*&trainY,
	cuMatrix<int>*&testY)
{
	/*read the train data and train label*/
	string filename;
	filename = "cifar-10-batches-bin/data_batch_";
	
	trainY = new cuMatrix<int>(50000, 1, 1);
	char str[25];
	
	for(int i = 1; i <= 5; i++)
	{
		sprintf(str, "%d", i);
		string name = filename + string(str) + ".bin";
		read_batch(name, trainX, trainY);
	}
	trainX.toGpu();
	trainY->toGpu();

	/*read the test data and test labels*/

	filename = "cifar-10-batches-bin/test_batch.bin";
	testY = new cuMatrix<int>(10000, 1, 1);
	read_batch(filename, testX,testY);
	testX.toGpu();
	testY->toGpu();
}