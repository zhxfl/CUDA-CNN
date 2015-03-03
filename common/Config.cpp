#include "Config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string>
using namespace std;


bool Config::get_word_bool(string &str, string name){

	size_t pos = str.find(name);    
	int i = pos + 1;
	bool res = true;
	while(1){
		if(i == str.length()) break;
		if(str[i] == ';') break;
		++ i;
	}
	string sub = str.substr(pos, i - pos + 1);
	if(sub[sub.length() - 1] == ';'){
		string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
		if(!content.compare("true")) res = true;
		else res = false;
	}
	str.erase(pos, i - pos + 1);
	return res;
}

int Config::get_word_int(string &str, string name){

		size_t pos = str.find(name);    
		int i = pos + 1;
		int res = 1;
		while(1){
			if(i == str.length()) break;
			if(str[i] == ';') break;
			++ i;
		}
		string sub = str.substr(pos, i - pos + 1);
		if(sub[sub.length() - 1] == ';'){
			string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
			res = atoi(content.c_str());
		}
		str.erase(pos, i - pos + 1);
		return res;
}

double Config::get_word_double(string &str, string name){

		size_t pos = str.find(name);    
		int i = pos + 1;
		double res = 0.0;
		while(1){
			if(i == str.length()) break;
			if(str[i] == ';') break;
			++ i;
		}
		string sub = str.substr(pos, i - pos + 1);
		if(sub[sub.length() - 1] == ';'){
			string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
			res = atof(content.c_str());
		}
		str.erase(pos, i - pos + 1);
		return res;
}

string Config::get_word_type(string &str, string name){

		size_t pos = str.find(name);    
		int i = pos + 1;
		int res = 0;
		while(1){
			if(i == str.length()) break;
			if(str[i] == ';') break;
			++ i;
		}
		string sub = str.substr(pos, i - pos + 1);
		string content;
		if(sub[sub.length() - 1] == ';'){
			content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
		}
		str.erase(pos, i - pos + 1);
		return content;
}

void Config:: get_layers_config(string &str){
		vector<string> layers;
		if(str.empty()) return;
		int head = 0;
		int tail = 0;
		while(1){
			if(head == str.length()) break;
			if(str[head] == '['){
				tail = head + 1;
				while(1){
					if(tail == str.length()) break;
					if(str[tail] == ']') break;
					++ tail;
				}
				string sub = str.substr(head, tail - head + 1);
				if(sub[sub.length() - 1] == ']'){
					sub.erase(sub.begin() + sub.length() - 1);
					sub.erase(sub.begin());
					layers.push_back(sub);
				}
				str.erase(head, tail - head + 1);
			}else ++ head;
		}
		for(int i = 0; i < layers.size(); i++){
			string type = get_word_type(layers[i], "LAYER");
			if(type == string("CONV")) {
				int ks = get_word_int(layers[i], "KERNEL_SIZE");
				int ka = get_word_int(layers[i], "KERNEL_AMOUNT");
				int pd = get_word_int(layers[i], "PADDING");

				double wd = get_word_double(layers[i], "WEIGHT_DECAY");
				m_conv.push_back(new ConfigConv(ks, pd, ka, wd));
				printf("\n\n********conv layer********\n");
				printf("KERNEL_SIZE   : %d\n", ks);
				printf("KERNEL_AMOUNT : %d\n", ka);
				printf("PADDING       : %d\n", pd);
				printf("WEIGHT_DECAY  : %lf\n", wd);
			}
			else if(type == string("POOLING"))
			{
				int size = get_word_int(layers[i], "SIZE");
				int skip = get_word_int(layers[i], "SKIP");
				m_pooling.push_back(new ConfigPooling(size, skip));
				printf("\n\n********pooling layer********\n");
				printf("size : %d\n",  size);
				printf("skip : %d\n", skip);
			}
			else if(type == string("FC"))
			{
				int hn = get_word_int(layers[i], "NUM_FULLCONNECT_NEURONS");
				double wd = get_word_double(layers[i], "WEIGHT_DECAY");
				double drop = get_word_double(layers[i], "DROPCONNECT_RATE");
				m_fc.push_back(new ConfigFC(hn, wd, drop));

				printf("\n\n********Full Connect Layer********\n");
				printf("NUM_FULLCONNECT_NEURONS : %d\n", hn);
				printf("WEIGHT_DECAY            : %lf\n", wd);
				printf("DROPCONNECT_RATE        : %lf\n", drop);
			}
			else if(type == string("SOFTMAX"))
			{
				int numClasses = get_word_int(layers[i], "NUM_CLASSES");
				double weightDecay = get_word_double(layers[i], "WEIGHT_DECAY");
				m_softMax.push_back(new ConfigSoftMax(numClasses, weightDecay));

				printf("\n\n********SoftMax Layer********\n");
				printf("NUM_CLASSES  : %d\n", numClasses);
				printf("WEIGHT_DECAY : %lf\n", weightDecay);
			}
		}
}

void Config::init(std::string path)
{
	printf("\n\n*******************CONFIG*******************\n");
	/*read the string from file "Config.txt"*/
	/*delete the comment and spaces*/
	m_configStr = read_2_string(path);
	deleteComment();
	deleteSpace();

	/*IS_GRADIENT_CHECKING*/
	bool is_gradient_checking = get_word_bool(m_configStr, "IS_GRADIENT_CHECKING");
	m_isGrandientChecking = new ConfigGradient(is_gradient_checking);
	printf("Is Grandient Checking : %d\n", is_gradient_checking);

	/*BATCH_SIZE*/
	int batch_size = get_word_int(m_configStr, "BATCH_SIZE");
	m_batchSize = new ConfigBatchSize(batch_size);
	printf("batch Size            : %d\n", batch_size);

	/*NON_LINEARITY*/
	string non_linearity = get_word_type(m_configStr, "NON_LINEARITY");
	m_nonLinearity = new ConfigNonLinearity(non_linearity);
	printf("non_linearity         : %s\n", non_linearity.c_str());

	/*CHANNELS*/
	int channels = get_word_int(m_configStr, "CHANNELS");
	m_channels = new ConfigChannels(channels);
	printf("channels              : %d\n", channels);

	/*crop*/
	int crop = get_word_int(m_configStr, "CROP");
	m_crop = new ConfigCrop(crop);
	printf("crop                  : %d\n", crop);

	/*scale*/
	double scale = get_word_double(m_configStr, "SCALE");
	m_scale = new ConfigScale(scale);
	printf("scale                 : %lf\n", scale);

	/*rotation*/
	double rotation = get_word_double(m_configStr, "ROTATION");
	m_rotation = new ConfigRotation(rotation);
	printf("rotation              : %lf\n", rotation);

	/*distortion*/
	double distortion = get_word_double(m_configStr, "DISTORTION");
	m_distortion = new ConfigDistortion(distortion);
	printf("distortion            : %lf\n", distortion);

	/*ImageShow*/
	bool imageShow = get_word_bool(m_configStr, "SHOWIMAGE");
	m_imageShow = new ConfigImageShow(imageShow);
	printf("imageShow             : %d\n", imageShow);

	/*Horizontal*/
	bool horizontal = get_word_bool(m_configStr, "HORIZONTAL");
	m_horizontal = new ConfigHorizontal(horizontal);
	printf("HORIZONTAL            : %d\n", horizontal);

	/*Combine Feature Maps*/
	int cfm = get_word_int(m_configStr, "COMBINE_FEATRUE_MAPS");
	m_cfm = new ConfigCombineFeatureMaps(cfm);
	printf("combine feature maps  : %d\n", cfm);

	/*Layers*/
	get_layers_config(m_configStr);
	printf("\n\n\n");
}

void Config::deleteSpace()
{
	if(m_configStr.empty()) return;
	int i = 0;
	while(1){
		if(i == m_configStr.length()) break;
		if(m_configStr[i] == '\t' || m_configStr[i] == '\n' || m_configStr[i] == ' '){
			m_configStr.erase(m_configStr.begin() + i);
		}else ++ i;
	}
}

void Config::deleteComment()
{
	size_t pos1, pos2;
	do 
	{
		pos1 = m_configStr.find("#");
		if(pos1 == std::string::npos)
			break;
		m_configStr.erase(pos1, 1);
		pos2 = m_configStr.find("#");
		m_configStr.erase(pos1, pos2 - pos1 + 1);
	} while (pos2 != std::string::npos);
}

string 
	Config::read_2_string(string File_name){
		char *pBuf;
		FILE *pFile = NULL;   
		if(!(pFile = fopen(File_name.c_str(),"r"))){
			printf("Can not find this file.");
			return 0;
		}
		//move pointer to the end of the file
		fseek(pFile, 0, SEEK_END);
		//Gets the current position of a file pointer.offset 
		int len = ftell(pFile);
		pBuf = new char[len];
		//Repositions the file pointer to the beginning of a file
		rewind(pFile);
		fread(pBuf, 1, len, pFile);
		fclose(pFile);
		string res = pBuf;
		return res;
}
