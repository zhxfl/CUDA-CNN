#ifndef __CONFIG_H__
#define __CONFIG_H__


#include <string>
#include <vector>
#include "util.h"
#include <time.h>

class ConfigGradient
{
public:
	ConfigGradient(bool isGradientChecking)
	{
		m_IsGradientChecking = isGradientChecking;
	}
	bool getValue(){return m_IsGradientChecking;};
private:
	bool m_IsGradientChecking;
};



class ConfigImageShow
{
public:
	ConfigImageShow(bool imageShow)
	{
		m_imageShow = imageShow;
	}
	bool getValue(){return m_imageShow;};
private:
	bool m_imageShow;
};

class ConfigCrop
{
public:
	ConfigCrop(int crop)
	{
		m_crop = crop;
	}
	int getValue(){return m_crop;};
private:
	int m_crop;
};

class ConfigScale
{
public:
	ConfigScale(double scale):m_scale(scale){}
	double getValue(){return m_scale;}
private:
	double m_scale;
};

class ConfigWhiteNoise
{
public:
	ConfigWhiteNoise(double stdev):m_stdev(stdev){}
	double getValue(){return m_stdev;}
private:
	double m_stdev;
};

class ConfigTestEpoch
{
public:
	ConfigTestEpoch(int testEpoch)
	{
		m_testEpoch = testEpoch;
	}
	int getValue(){return m_testEpoch;}
private:
	int m_testEpoch;
};

class ConfigRotation
{
public:
	ConfigRotation(double rotation)
	{
		m_rotation = rotation;
	}
	double getValue()
	{
		return m_rotation;
	}
private:
	double m_rotation;

};

class ConfigDistortion
{
public:
	ConfigDistortion(double distortion)
	{
		m_distortion = distortion;
	}
	double getValue()
	{
		return m_distortion;
	}
private:
	double m_distortion;
};

class ConfigBatchSize
{
public:
	ConfigBatchSize(int batchSize)
	{
		m_batchSize = batchSize;
	}
	int getValue(){
		return m_batchSize;
	}
private:
	int m_batchSize;
};

class ConfigNonLinearity
{
public:
	ConfigNonLinearity(std::string poolMethod)
	{
		if(poolMethod == std::string("NL_SIGMOID")){
			m_nonLinearity = NL_SIGMOID;
		}else if(poolMethod == std::string("NL_TANH")){
			m_nonLinearity = NL_TANH;
		}else if(poolMethod == std::string("NL_RELU")){
			m_nonLinearity = NL_RELU;
		}
		else{
			m_nonLinearity = -1;
		}
	}
	int getValue(){return m_nonLinearity;}
private:
	int m_nonLinearity;
};



class ConfigChannels
{
public:
	ConfigChannels(int channels):m_channels(channels){};
	int m_channels;

	int getValue(){return m_channels;};
};

class ConfigBase
{
public:
	std::string m_name;
	std::string m_input;
	std::string m_subInput;
	std::vector<ConfigBase*> m_next;
	std::string m_type;
	double m_initW;
	int m_nonLinearity; 
	std::string m_initType;
	bool isGaussian(){
		return m_initType == std::string("Gaussian");
	}
	bool hasSubInput(){
		return m_subInput != std::string("NULL");
	}
	bool isBranchLayer(){
		return m_type == std::string("BRANCHLAYER");
	}
};

class ConfigLRN : public ConfigBase{
public:
	ConfigLRN(std::string name, 
		std::string input,
		std::string subInput,
		std::string type,
		double k, int n, 
		double alpha, double belta, 
		int nonLinearity):m_k(k), m_n(n), m_alpha(alpha), m_belta(belta){
		  m_nonLinearity = nonLinearity;
		  m_name  = name;
		  m_input = input; 
		  m_subInput = subInput;
		  m_type = type;
	  }
	double m_k;
	double m_alpha;
	double m_belta;
	int m_n;
};


class ConfigBranchLayer: public ConfigBase{
public:
	ConfigBranchLayer(std::string name, 
		std::string input,
		std::string subInput,
		std::string type,
		std::vector<std::string>&outputs){
			m_name = name;
			m_input = input;
			m_subInput = subInput;
			m_type = type;
			m_outputs = outputs;
	}
	std::vector<std::string>m_outputs;
};


class ConfigCombineLayer:public ConfigBase{
public:
	ConfigCombineLayer(std::string name, 
		std::vector<std::string> inputs,
		std::string subInput,
		std::string type){
			m_name = name;
			m_inputs = inputs;
			m_subInput = subInput;
			m_type = type;
			m_input = std::string("NULL");
	}
	std::vector<std::string>m_inputs;
};


class ConfigData:public ConfigBase{
public:
	ConfigData(std::string name,
		std::string type){
		m_name = name;
		m_type = type;
		m_input = std::string("NULL");
	}
};

class ConfigConv : public ConfigBase
{
public:
	ConfigConv(std::string name, std::string input,
		std::string subInput, std::string type,
		int kernelSize, int padding, int amount,
		double weightDecay, int cfm, double initW, std::string initType,
		int non_linearity){
		m_kernelSize = kernelSize;
		m_padding = padding;
		m_amount = amount;
		m_weightDecay = weightDecay;
		m_name = name;
		m_input = input;
		m_subInput = subInput;
		m_cfm = cfm;
		m_type = type;
		m_initW = initW;
		m_nonLinearity = non_linearity;
		m_initType = initType;
	}
	int m_cfm;
	int m_kernelSize;
	int m_padding;
	int m_amount;
	double m_weightDecay;
};


class ConfigLocal : public ConfigBase
{
public:
	ConfigLocal(std::string name,
		std::string input,
		std::string subInput,
		std::string type,
		int kernelSize,
		double weightDecay, double initW, std::string initType,
		int non_linearity){
			m_kernelSize = kernelSize;
			m_weightDecay = weightDecay;
			m_name = name;
			m_input = input;
			m_subInput = subInput;
			m_type = type;
			m_initW = initW;
			m_nonLinearity = non_linearity;
			m_initType = initType;
	}
	int m_kernelSize;
	double m_weightDecay;
};

/*
[
	LAYER = NIN;
	NAME  = nin1;
	INPUT = conv3;
	WEIGHT_DECAY = 1e-6;
]
*/
class ConfigNIN : public ConfigBase{
public:
	ConfigNIN(std::string name, 
		std::string input, 
		std::string subInput,
		std::string type,
		double weightDecay){
		m_name = name;
		m_input= input;
		m_subInput = subInput;
		m_type = type;
		m_weightDecay = weightDecay;
	}
	double m_weightDecay;
};

class ConfigPooling : public ConfigBase
{
public:
	ConfigPooling(std::string name,
		std::string input,
		std::string subInput,
		std::string type,
		std::string poolingType,
		int size, int skip,
		int non_linearity){
		m_size = size;
		m_skip = skip;
		m_name = name;
		m_input = input;
		m_subInput = subInput;
		m_type = type;
		m_nonLinearity = non_linearity;
		m_poolingType = poolingType;
	}
	int m_size;
	int m_skip;
	std::string m_poolingType;
};

class ConfigFC : public ConfigBase
{
public:
	ConfigFC(std::string name,
		std::string input,
		std::string subInput,
		std::string type,
		int numFullConnectNeurons,
		double weightDecay,
		double dropoutRate,
		double initW, 
		std::string initType, 
		int non_linearity)
	{
		m_numFullConnectNeurons = numFullConnectNeurons;
		m_weightDecay = weightDecay;
		m_dropoutRate = dropoutRate;
		m_name = name;
		m_input = input;
		m_subInput = subInput;
		m_type = type;
		m_initW = initW;
		m_nonLinearity = non_linearity;
		m_initType = initType;
	}
	int m_numFullConnectNeurons;
	double m_weightDecay;
	double m_dropoutRate;
};

class ConfigSoftMax : public ConfigBase
{
public:
	int m_numClasses;
	double m_weightDecay;
	ConfigSoftMax(std::string name,
		std::string input,
		std::string subInput,
		std::string type, 
		int numClasses, 
		double weightDecay, 
		double initW, 
		std::string initType,
		int non_linearity)
	{
		m_numClasses = numClasses;
		m_weightDecay = weightDecay;
		m_name = name;
		m_input = input;
		m_subInput = subInput;
		m_type = type;
		m_initW = initW;
		m_nonLinearity = non_linearity;
		m_initType = initType;
	}
};

class ConfigHorizontal
{
public:
	ConfigHorizontal(int horizontal)
	{
		m_horizontal = horizontal;
	}
	int getValue(){
		return m_horizontal;
	}
private:
	int m_horizontal;
};


class Config
{
public:
	void setMomentum(double _momentum){
		momentum = _momentum;
	}
	double getMomentum(){
		return momentum;
	}
	void setLrate(double _lrate){
		lrate = _lrate;
	}
	double getLrate(){
		return lrate;
	}
	void initPath(std::string path){
		m_path = path;
		init(m_path);
	}
	static Config* instance(){
		static Config* config = new Config();
		return config;
	}

	void clear(){

		delete  m_nonLinearity;
		delete  m_isGrandientChecking;
		delete  m_batchSize;
		delete  m_channels;

		delete m_crop;
		delete m_scale;
		delete m_rotation;
		delete m_distortion;
		delete m_imageShow;
		delete m_horizontal;
	}

	bool getImageShow(){
		return m_imageShow->getValue();}

	bool getIsGradientChecking(){
		return m_isGrandientChecking->getValue();}

	int getBatchSize(){
		return m_batchSize->getValue();}

	int getChannels(){
		return m_channels->getValue();
	}

	int getCrop(){
		return m_crop->getValue();
	}

	int getHorizontal(){
		return m_horizontal->getValue();
	}

	double getScale(){
		return m_scale->getValue();
	}

	double getRotation(){
		return m_rotation->getValue();
	}

	double getDistortion(){
		return m_distortion->getValue();
	}

	const std::vector<ConfigBase*> getFirstLayers(){
		return m_firstLayers;
	}

	ConfigBase* getLayerByName(std::string name){
		if(m_layerMaps.find(name) != m_layerMaps.end()){
			return m_layerMaps[name];
		}
		else{
			char logStr[1024];
			sprintf(logStr, "layer %s does not exit\n", name.c_str());
			LOG(logStr, "Result/log.txt");
			exit(0);
		}
	}

	void insertLayerByName(std::string name, ConfigBase* layer){
		if(m_layerMaps.find(name) == m_layerMaps.end()){
			m_layerMaps[name] = layer;
		}
		else {
			char logStr[1024];
			sprintf(logStr, "layer %s exit\n", name.c_str());LOG(logStr, "Result/log.txt");
			exit(0);
		}
	}

	void setImageSize(int imageSize){
		m_imageSize = imageSize;
	}

	int getImageSize(){
		return m_imageSize;
	}

	int getTestEpoch(){
		return m_test_epoch->getValue();
	}

	double getWhiteNoise(){
		return m_white_noise->getValue();
	}

	int getClasses(){
		return m_classes;
	}
	void setTraining(bool isTrainning){training = isTrainning;}
	bool isTraining(){return training;}

private:
	void deleteComment();
	void deleteSpace();
	bool get_word_bool(std::string &str, std::string name);
	std::string get_word_type(std::string &str, std::string name);
	double get_word_double(std::string &str, std::string name);
	int get_word_int(std::string &str, std::string name);
	std::string read_2_string(std::string File_name);
	void get_layers_config(std::string &str);
	std::vector<std::string> get_name_vector(std::string &str, std::string name);

	void init(std::string path);
	std::string m_configStr;
	std::string m_path;

	std::map<std::string, ConfigBase*>m_layerMaps;
	std::vector<ConfigBase*>m_firstLayers;



	ConfigNonLinearity       *m_nonLinearity;
	ConfigGradient           *m_isGrandientChecking;
	ConfigBatchSize          *m_batchSize;
	ConfigChannels           *m_channels;

	ConfigCrop               *m_crop;
	ConfigScale              *m_scale;
	ConfigRotation           *m_rotation;
	ConfigDistortion         *m_distortion;
	ConfigImageShow          *m_imageShow;
	ConfigHorizontal         *m_horizontal;
	ConfigTestEpoch          *m_test_epoch;
	ConfigWhiteNoise         *m_white_noise;
	double momentum;
	double lrate;
	int m_imageSize;
	int m_classes;
	bool training;
};

#endif
