#include "Config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string>
using namespace std;


bool Config::get_word_bool(string &str, string name){
	size_t pos = str.find(name+"=");   
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
		size_t pos = str.find(name+"=");  
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
		size_t pos = str.find(name+"=");    
		if(pos == str.npos){
			return "NULL";
		}

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

std::vector<string> Config::get_name_vector(string &str, string name){
	std::vector<std::string>result;
	
	size_t pos = str.find(name+"=");    
	if(pos == str.npos){
		return result;
	}

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

	while(content.size()){
		size_t pos = content.find(',');
		if(pos == str.npos){
			result.push_back(content);
			break;
		}else{
			result.push_back(content.substr(0, pos));
			content.erase(0, pos + 1);
		}
	}

	return result;
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
		std::string name = get_word_type(layers[i], "NAME");
		std::string input = get_word_type(layers[i], "INPUT");
		std::string subInput = get_word_type(layers[i], "SUBINPUT");

		ConfigBase* layer;
		if(type == string("CONV")) {
			int ks = get_word_int(layers[i], "KERNEL_SIZE");
			int ka = get_word_int(layers[i], "KERNEL_AMOUNT");
			int pd = get_word_int(layers[i], "PADDING");
			int cfm= get_word_int(layers[i], "COMBINE_FEATRUE_MAPS");
			double initW = get_word_double(layers[i], "initW");
			std::string initType = get_word_type(layers[i], "initType");

			double wd = get_word_double(layers[i], "WEIGHT_DECAY");
			string non_linearity = get_word_type(layers[i], "NON_LINEARITY");
			m_nonLinearity = new ConfigNonLinearity(non_linearity);

			layer = new ConfigConv(name, input, subInput, type, ks, pd, ka, wd, cfm,
				initW, initType, m_nonLinearity->getValue());
			char logStr[256];
			sprintf(logStr,"\n\n********conv layer********\n");        LOG(logStr, "Result/log.txt");
			sprintf(logStr, "NAME          : %s\n", name.c_str());     LOG(logStr, "Result/log.txt");
			sprintf(logStr, "INPUT         : %s\n", input.c_str());    LOG(logStr, "Result/log.txt");
			sprintf(logStr, "SUBINPUT      : %s\n", subInput.c_str()); LOG(logStr, "Result/log.txt");
			sprintf(logStr, "KERNEL_SIZE   : %d\n", ks);               LOG(logStr, "Result/log.txt");
			sprintf(logStr, "KERNEL_AMOUNT : %d\n", ka);               LOG(logStr, "Result/log.txt");
			sprintf(logStr, "CFM           : %d\n", cfm);              LOG(logStr, "Result/log.txt");
			sprintf(logStr, "PADDING       : %d\n", pd);               LOG(logStr, "Result/log.txt");
			sprintf(logStr, "WEIGHT_DECAY  : %lf\n", wd);              LOG(logStr, "Result/log.txt");
			sprintf(logStr, "initW         : %lf\n", initW);           LOG(logStr, "Result/log.txt");
			sprintf(logStr, "non_linearity : %s\n", non_linearity.c_str()); LOG(logStr, "Result/log.txt");
		}
		else if(type == string("DATA")){
			layer = new ConfigData(name, type);
			char logStr[256];
			sprintf(logStr, "\n\n********data Layer********\n"); LOG(logStr, "Result/log.txt");
			sprintf(logStr, "NAME          : %s\n", name.c_str()); LOG(logStr, "Result/log.txt");
		}
		else if(type == string("BRANCHLAYER")){
			std::vector<std::string>outputs = get_name_vector(layers[i], "OUTPUTS");
			layer = new ConfigBranchLayer(name, input, subInput, type, outputs);

			char logStr[256];
			sprintf(logStr, "\n\n********branch layer********\n");LOG(logStr, "Result/log.txt");
			sprintf(logStr, "NAME          : %s\n", name.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "INPUT         : %s\n", input.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "SUBINPUT      : %s\n", subInput.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "OUTPUTS       :");LOG(logStr, "Result/log.txt");
			for(int i = 0; i < outputs.size(); i++){
				sprintf(logStr, "%s,", outputs[i].c_str());LOG(logStr, "Result/log.txt");
			}sprintf(logStr, "\n");LOG(logStr, "Result/log.txt");
		}
		else if(type == string("COMBINELAYER")){
			std::vector<std::string>inputs = get_name_vector(layers[i], "INPUTS");
			layer = new ConfigCombineLayer(name, inputs, subInput, type);

			char logStr[256];
			sprintf(logStr, "\n\n********combine Layer********\n");LOG(logStr, "Result/log.txt");
			sprintf(logStr, "NAME          : %s\n", name.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "SUBINPUT      : %s\n", subInput.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "INPUTS        :");LOG(logStr, "Result/log.txt");

			for(int i = 0; i < inputs.size(); i++){
				sprintf(logStr, "%s,", inputs[i].c_str());LOG(logStr, "Result/log.txt");
			}sprintf(logStr, "\n");LOG(logStr, "Result/log.txt");
		}
		else if(type == string("NIN")){
			double wd = get_word_double(layers[i], "WEIGHT_DECAY");
			layer = new ConfigNIN(name, input, subInput, type, wd);
			
			char logStr[256];
			sprintf(logStr, "\n\n********NIN layer********\n");LOG(logStr, "Result/log.txt");
			sprintf(logStr, "NAME          : %s\n", name.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "INPUT         : %s\n", input.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "SUBINPUT      : %s\n", subInput.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "WEIGHT_DECAY  : %lf\n", wd);LOG(logStr, "Result/log.txt");
		}
		else if(type == string("POOLING"))
		{
			int size = get_word_int(layers[i], "SIZE");
			int skip = get_word_int(layers[i], "SKIP");
			std::string poolingType = get_word_type(layers[i], "POOLINGTYPE");
			string non_linearity = get_word_type(layers[i], "NON_LINEARITY");

			m_nonLinearity = new ConfigNonLinearity(non_linearity);

			layer = new ConfigPooling(name, input, subInput, type, poolingType, size, skip, m_nonLinearity->getValue());
			char logStr[256];
			sprintf(logStr, "\n\n********pooling layer********\n"); LOG(logStr, "Result/log.txt");
			sprintf(logStr, "NAME          : %s\n", name.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "INPUT         : %s\n", input.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "POOLINGTYPE   : %s\n", poolingType.c_str()); LOG(logStr, "Result/log.txt");
			sprintf(logStr, "SUBINPUT      : %s\n", subInput.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "size          : %d\n", size);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "skip          : %d\n", skip);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "non_linearity : %s\n", non_linearity.c_str());LOG(logStr, "Result/log.txt");
		}
		else if(string("LOCAL") == type){
			int ks = get_word_int(layers[i], "KERNEL_SIZE");
			double initW = get_word_double(layers[i], "initW");
			double wd = get_word_double(layers[i], "WEIGHT_DECAY");
			string non_linearity = get_word_type(layers[i], "NON_LINEARITY");
			std::string initType = get_word_type(layers[i], "initType");
			m_nonLinearity = new ConfigNonLinearity(non_linearity);

			layer = new ConfigLocal(name, input, subInput, type, ks, wd,
				initW, initType, m_nonLinearity->getValue());
			
			char logStr[256];
			sprintf(logStr, "\n\n********local connect layer********\n");LOG(logStr, "Result/log.txt");
			sprintf(logStr, "NAME          : %s\n", name.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "INPUT         : %s\n", input.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "SUBINPUT      : %s\n", subInput.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "KERNEL_SIZE   : %d\n", ks);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "WEIGHT_DECAY  : %lf\n", wd);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "initW         : %lf\n", initW);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "non_linearity : %s\n", non_linearity.c_str());LOG(logStr, "Result/log.txt");
		}
		else if(string("LRN") == type){
			double lrn_k     = get_word_double(layers[i], "LRN_K");
			int    lrn_n     = get_word_int(layers[i], "LRN_N");
			double lrn_alpha = get_word_double(layers[i], "LRN_ALPHA");
			double lrn_belta = get_word_double(layers[i], "LRN_BALTA");
			string non_linearity = get_word_type(layers[i], "NON_LINEARITY");
			m_nonLinearity = new ConfigNonLinearity(non_linearity);

			layer = new ConfigLRN(name, input, subInput, type, lrn_k, lrn_n, lrn_alpha, lrn_belta, 
				m_nonLinearity->getValue());

			char logStr[256];
			sprintf(logStr, "\n\n********local Response Normalization layer********\n");LOG(logStr, "Result/log.txt");
			sprintf(logStr, "NAME          : %s\n", name.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "INPUT         : %s\n", input.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "SUBINPUT      : %s\n", subInput.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "lrn_k         : %lf\n", lrn_k);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "lrn_n         : %d\n",  lrn_n);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "lrn_alpha     : %lf\n", lrn_alpha);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "lrn_belta     : %lf\n", lrn_belta);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "non_linearity : %s\n", non_linearity.c_str());LOG(logStr, "Result/log.txt");
		}
		else if(type == string("FC"))
		{
			int hn = get_word_int(layers[i], "NUM_FULLCONNECT_NEURONS");
			double wd = get_word_double(layers[i], "WEIGHT_DECAY");
			double drop = get_word_double(layers[i], "DROPCONNECT_RATE");
			double initW= get_word_double(layers[i], "initW");
			string non_linearity = get_word_type(layers[i], "NON_LINEARITY");
			std::string initType = get_word_type(layers[i], "initType");
			m_nonLinearity = new ConfigNonLinearity(non_linearity);

			layer = new ConfigFC(name, input, subInput, type, hn, wd,
				drop, initW, initType, m_nonLinearity->getValue());

			char logStr[256];
			sprintf(logStr, "\n\n********Full Connect Layer********\n");LOG(logStr, "Result/log.txt");
			sprintf(logStr, "NAME                    : %s\n", name.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "INPUT                   : %s\n", input.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "SUBINPUT                : %s\n", subInput.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "NUM_FULLCONNECT_NEURONS : %d\n", hn);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "WEIGHT_DECAY            : %lf\n", wd);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "DROPCONNECT_RATE        : %lf\n", drop);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "initW                   : %lf\n", initW);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "non_linearity           : %s\n", non_linearity.c_str());LOG(logStr, "Result/log.txt");
		}
		else if(type == string("SOFTMAX"))
		{
			int numClasses = get_word_int(layers[i], "NUM_CLASSES");
			double weightDecay = get_word_double(layers[i], "WEIGHT_DECAY");
			double initW= get_word_double(layers[i], "initW");
			string non_linearity = get_word_type(layers[i], "NON_LINEARITY");
			std::string initType = get_word_type(layers[i], "initType");
			m_nonLinearity = new ConfigNonLinearity(non_linearity);
			layer = new ConfigSoftMax(name, input, subInput, type, numClasses, weightDecay, 
				initW, initType, m_nonLinearity->getValue());
			m_classes = numClasses;

			char logStr[256];
			sprintf(logStr, "\n\n********SoftMax Layer********\n");LOG(logStr, "Result/log.txt");
			sprintf(logStr, "NAME         : %s\n", name.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "INPUT        : %s\n", input.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "SUBINPUT     : %s\n", subInput.c_str());LOG(logStr, "Result/log.txt");
			sprintf(logStr, "NUM_CLASSES  : %d\n", numClasses);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "WEIGHT_DECAY : %lf\n", weightDecay);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "initW        : %lf\n", initW);LOG(logStr, "Result/log.txt");
			sprintf(logStr, "non_linearity: %s\n", non_linearity.c_str());LOG(logStr, "Result/log.txt");
		}

		insertLayerByName(name, layer);
		if(type == std::string("DATA")){
			m_firstLayers.push_back(layer);
		}
		else{
			if(layer->m_type == string("COMBINELAYER")){
				ConfigCombineLayer *cl = static_cast<ConfigCombineLayer*>(layer);
				for(int i = 0; i < cl->m_inputs.size(); i++){
					ConfigBase* preLayer = getLayerByName(cl->m_inputs[i]);
					preLayer->m_next.push_back(layer);
				}
			}
			else{
				ConfigBase* preLayer = getLayerByName(layer->m_input);
				preLayer->m_next.push_back(layer);
			}
		}
	}
}

void Config::init(std::string path)
{
	char logStr[256];
	sprintf(logStr, "\n\n*******************CONFIG*******************\n");
	LOG(logStr, "Result/log.txt");

	/*read the string from file "Config.txt"*/
	/*delete the comment and spaces*/
	m_configStr = read_2_string(path);
	deleteComment();
	deleteSpace();

	/*IS_GRADIENT_CHECKING*/
	bool is_gradient_checking = get_word_bool(m_configStr, "IS_GRADIENT_CHECKING");
	m_isGrandientChecking = new ConfigGradient(is_gradient_checking);
	sprintf(logStr, "Is Grandient Checking : %d\n", is_gradient_checking);
	LOG(logStr, "Result/log.txt");

	/*BATCH_SIZE*/
	int batch_size = get_word_int(m_configStr, "BATCH_SIZE");
	m_batchSize = new ConfigBatchSize(batch_size);
	sprintf(logStr, "batch Size            : %d\n", batch_size);
	LOG(logStr, "Result/log.txt");

	/*CHANNELS*/
	int channels = get_word_int(m_configStr, "CHANNELS");
	m_channels = new ConfigChannels(channels);
	sprintf(logStr, "channels              : %d\n", channels);
	LOG(logStr, "Result/log.txt");

	/*crop*/
	int crop = get_word_int(m_configStr, "CROP");
	m_crop = new ConfigCrop(crop);
	sprintf(logStr, "crop                  : %d\n", crop);
	LOG(logStr, "Result/log.txt");

	/*scale*/
	double scale = get_word_double(m_configStr, "SCALE");
	m_scale = new ConfigScale(scale);
	sprintf(logStr, "scale                 : %lf\n", scale);
	LOG(logStr, "Result/log.txt");

	/*rotation*/
	double rotation = get_word_double(m_configStr, "ROTATION");
	m_rotation = new ConfigRotation(rotation);
	sprintf(logStr, "rotation              : %lf\n", rotation);
	LOG(logStr, "Result/log.txt");

	/*distortion*/
	double distortion = get_word_double(m_configStr, "DISTORTION");
	m_distortion = new ConfigDistortion(distortion);
	sprintf(logStr, "distortion            : %lf\n", distortion);
	LOG(logStr, "Result/log.txt");

	/*ImageShow*/
	bool imageShow = get_word_bool(m_configStr, "SHOWIMAGE");
	m_imageShow = new ConfigImageShow(imageShow);
	sprintf(logStr, "imageShow             : %d\n", imageShow);
	LOG(logStr, "Result/log.txt");

	/*Horizontal*/
	bool horizontal = get_word_bool(m_configStr, "HORIZONTAL");
	m_horizontal = new ConfigHorizontal(horizontal);
	sprintf(logStr, "HORIZONTAL            : %d\n", horizontal);
	LOG(logStr, "Result/log.txt");

	/*Test Epoch*/
	int test_epoch = get_word_int(m_configStr, "TEST_EPOCH");
	m_test_epoch = new ConfigTestEpoch(test_epoch);
	sprintf(logStr, "Test_Epoch            : %d\n", test_epoch);
	LOG(logStr, "Result/log.txt");

	/*WHITE_NOISE*/
	double stdev = get_word_double(m_configStr, "WHITE_NOISE");
	m_white_noise = new ConfigWhiteNoise(stdev);
	sprintf(logStr, "White Noise           : %f\n", stdev);
	LOG(logStr, "Result/log.txt");

	/*Layers*/
	get_layers_config(m_configStr);
	sprintf(logStr, "\n\n\n");
	LOG(logStr, "Result/log.txt");
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
		char logStr[1025];
		if(!(pFile = fopen(File_name.c_str(),"r"))){
			sprintf(logStr, "Can not find this file.");
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
