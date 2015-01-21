#include "readCIFAR10Data.h"
#include <string>
#include <fstream>
#include <vector>
#include <vector>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
using namespace std;

#ifdef linux
#include <dirent.h>
#endif


#ifdef linux
void readChineseData(cuMatrixVector<double> &trainX,
		cuMatrixVector<double>&testX, cuMatrix<int>*&trainY,
		cuMatrix<int>*&testY) {

	int ImgSize = 28;
	DIR *dir;
	char train[256] = "chinese/train/";
	char test[256] = "chinese/test/";

	struct dirent *ptr;

	std::vector<string> train_type;
	std::vector<string> test_type;
	int train_num = 0, test_num = 0, i = 0;

	/*train*/
	if ((dir = opendir(train)) == NULL) {
		perror("Open dir error...");
		exit(1);
	}

	while ((ptr = readdir(dir)) != NULL) {
		if (strcmp(ptr->d_name, ".") && strcmp(ptr->d_name, "..")) {
			train_type.push_back(string(ptr->d_name));
		}
	}
	closedir(dir);

	/*test*/
	if ((dir = opendir(test)) == NULL) {
		perror("Open dir error...");
		exit(1);
	}

	while ((ptr = readdir(dir)) != NULL) {
		if (strcmp(ptr->d_name, ".") && strcmp(ptr->d_name, "..")) {
			test_type.push_back(string(ptr->d_name));
		}
	}
	closedir(dir);

	std::vector<std::string> trainSet;
	std::vector<std::string> testSet;
	std::vector<int> trainLabel;
	std::vector<int> testLabel;

	for (int i = 0; i < train_type.size(); i++) {
		string Path = string(train) + train_type[i] + string("/");
		if ((dir = opendir(Path.c_str())) == NULL) {
			perror("Open dir error...");
			exit(1);
		}

		while ((ptr = readdir(dir)) != NULL) {
			if (strcmp(ptr->d_name, ".") && strcmp(ptr->d_name, "..")) {
				trainSet.push_back(Path + string(ptr->d_name));
				trainLabel.push_back(i);

				printf("%s\n", trainSet[trainSet.size() - 1].c_str());
			}
		}
		closedir(dir);
	}

	for (int i = 0; i < test_type.size(); i++) {
		string Path = string(test) + test_type[i] + string("/");
		if ((dir = opendir(Path.c_str())) == NULL) {
			perror("Open dir error...");
			exit(1);
		}

		while ((ptr = readdir(dir)) != NULL) {
			if (strcmp(ptr->d_name, ".") && strcmp(ptr->d_name, "..")) {
				testSet.push_back(Path + string(ptr->d_name));
				testLabel.push_back(i);
				printf("%s\n", testSet[testSet.size() - 1].c_str());
			}
		}
		closedir(dir);
	}

	srand(time(0));
	for(int i = 0; i < 500; i++)
	{
		int x = rand() % trainSet.size();
		int y = rand() % trainSet.size();

		swap(trainSet[x], trainSet[y]);
		swap(trainLabel[x], trainLabel[y]);

		x = rand() % testSet.size();
		y = rand() % testSet.size();
		swap(testSet[x], testSet[y]);
		swap(testLabel[x], testLabel[y]);
	}

	printf("\n\n****************Read Data Set****************\n");
	printf("train dataset size = %d\n", trainSet.size());
	printf("test  dataset size = %d\n", testSet.size());
	printf("*********************************************\n\n\n");

	trainY = new cuMatrix<int>(trainLabel.size(), 1, 1);
	testY  = new cuMatrix<int>(testLabel.size(), 1, 1);

	for(int i = 0; i < trainSet.size(); i++){
		cv::Mat img = cv::imread(trainSet[i], 0);
		cv::resize(img, img, cv::Size(ImgSize, ImgSize));
		//cv::imshow("1", img);
		//cv::waitKey(0);

		cuMatrix<double>* m = new cuMatrix<double>(ImgSize, ImgSize, 1);
		for(int r = 0; r < ImgSize; r++){
			for(int c = 0; c < ImgSize; c++){
				double val = img.at<uchar>(r, c);
				//val = (256.0 - val) / 256.0;
				val = (256.0 - val) / 256.0;
				m->set(r, c, 0, val);
			}
		}

		m->toGpu();
		trainX.push_back(m);
		trainY->set(i, 0, 0, trainLabel[i]);
//		showImg(trainX[i], 1);
//		cv::waitKey(0);

	}
	trainX.toGpu();
	trainY->toGpu();

	for(int i = 0; i < testSet.size(); i++){
		cv::Mat img = cv::imread(testSet[i], 0);
		cv::resize(img, img, cv::Size(ImgSize, ImgSize));

		cuMatrix<double>* m = new cuMatrix<double>(ImgSize, ImgSize, 1);
		for (int r = 0; r < ImgSize; r++) {
			for (int c = 0; c < ImgSize; c++) {
				double val = img.at<char>(r, c);
				val = (256.0 - val) / 256.0;
				m->set(r, c, 0, val);
			}
		}
		m->toGpu();
		testX.push_back(m);
		testY->set(i, 0, 0, testLabel[i]);
	}
	testX.toGpu();
	testY->toGpu();
}
#endif
