#include "cuMatrix.h"
#include "readMnistData.h"
#include <string>
#include <fstream>
#include <vector>
#include <cuda_runtime_api.h>
#include "util.h"
#include <vector>
#include "cuMatrixVector.h"

int checkError(int x)
{
	int mark[] = {2426,3532,4129,4476,7080,8086,9075,10048,10800,10994,
		12000,12132,12830,14542,15766,16130,20652,21130,23660,23886,
		23911,25562,26504,26560,30792,33506,35310,36104,38700,39354,
		40144,41284,42616,43109,43454,45352,49960,51280,51442,53396,
		53806,53930,54264,56596,57744,58022,59915};
	for(int i = 0; i < 47; i++)
	{
		if(mark[i] == x)
			return true;
	}
	return false;
}

/*reverse the int*/
int ReverseInt (int i){
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int) ch1 << 24) | ((int)ch2 << 16) | ((int)ch3 << 8) | ch4;
}

/*read the train data*/
int read_Mnist(std::string filename, 
	cuMatrixVector<double>& vec,
	int num,
	int flag){
		/*read the data from file*/
		std::ifstream file(filename.c_str(), std::ios::binary);
		int id = 0;
		if (file.is_open()){
			int magic_number = 0;
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;
			file.read((char*) &magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);
			file.read((char*) &number_of_images,sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);
			if(number_of_images >= num){
				number_of_images = num;
			}
			else{
				printf("read_Mnist::number of images is overflow\n");
				exit(0);
			}
			file.read((char*) &n_rows, sizeof(n_rows));
			n_rows = ReverseInt(n_rows);
			file.read((char*) &n_cols, sizeof(n_cols));
			n_cols = ReverseInt(n_cols);
			for(int i = 0; i < number_of_images; ++i){
				cuMatrix<double>* tpmat = new cuMatrix<double>(n_rows, n_cols);
				for(int r = 0; r < n_rows; ++r){
					for(int c = 0; c < n_cols; ++c){
						unsigned char temp = 0;
						file.read((char*) &temp, sizeof(temp));
						tpmat->set(r, c, (double)temp * 2.0 / 255.0 - 1.0);
					}
				}
				tpmat->toGpu();
				if(!flag){
					if(!checkError(id)){
						vec.push_back(tpmat);
					}
					else {
						printf("train data %d\n", id);
					}
				}
				else {
					vec.push_back(tpmat);
				}
				id++;
			}
		}
		vec.toGpu();
		return vec.size();
}

/*read the lable*/
int read_Mnist_Label(std::string filename, 
	cuMatrix<double>* &mat,
	int flag){
		std::ifstream file(filename.c_str(), std::ios::binary);
		int id = 0;
		if (file.is_open()){
			int magic_number = 0;
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;
			file.read((char*) &magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);
			file.read((char*) &number_of_images,sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);

			int id = 0;
			for(int i = 0; i < number_of_images; ++i){
				unsigned char temp = 0;
				file.read((char*) &temp, sizeof(temp));
				if(!flag){
					if(!checkError(i)){
						mat->set(id, 0, temp);
						id++;
					}
					else {
						printf("train label %d\n", i);
					}
				}
				else {
					mat->set(i,0,(double)temp);
					id++;
				}
			}
			mat->toGpu();
			if(!flag)return id;
			else return number_of_images;
			return id;
		}
		return 0;
}

/*read trainning data and lables*/
int readMnistData(cuMatrixVector<double>& x,
	cuMatrix<double>*& y, 
	std::string xpath,
	std::string ypath, 
	int number_of_images,
	int flag){
		/*read MNIST iamge into cuMatrix*/
		int len = read_Mnist(xpath, x, number_of_images, flag);
		/*read MNIST label into cuMatrix*/
		y = new cuMatrix<double>(len, 1);
		int t = read_Mnist_Label(ypath, y, flag);
		return t;
}
