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
	ConfigScale(double scale)
	{
		m_scale = scale;
	}
	double getValue(){return m_scale;}
private:
	double m_scale;
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
		if(poolMethod == std::string("NL_SIGMOID"))
		{
			m_nonLinearity = NL_SIGMOID;
		}else if(poolMethod == std::string("NL_TANH"))
		{
			m_nonLinearity = NL_TANH;
		}else 
		{
			m_nonLinearity = NL_RELU;
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
	std::vector<ConfigBase*> m_next;
	std::string m_type;
};

class ConfigConv : public ConfigBase
{
public:
	ConfigConv(std::string name, std::string input, std::string type, int kernelSize, int padding, int amount, double weightDecay, int cfm){
		m_kernelSize = kernelSize;
		m_padding = padding;
		m_amount = amount;
		m_weightDecay = weightDecay;
		m_name = name;
		m_input = input;
		m_cfm = cfm;
		m_type = type;
	}
	int m_cfm;
	int m_kernelSize;
	int m_padding;
	int m_amount;
	double m_weightDecay;
};

class ConfigPooling : public ConfigBase
{
public:
	ConfigPooling(std::string name, std::string input, std::string type, int size, int skip){
		m_size = size;
		m_skip = skip;
		m_name = name;
		m_input = input;
		m_type = type;
	}
	int m_size;
	int m_skip;
};

class ConfigFC : public ConfigBase
{
public:
	ConfigFC(std::string name, std::string input, std::string type, int numFullConnectNeurons, double weightDecay,
		double dropoutRate)
	{
		m_numFullConnectNeurons = numFullConnectNeurons;
		m_weightDecay = weightDecay;
		m_dropoutRate = dropoutRate;
		m_name = name;
		m_input = input;
		m_type = type;
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
	ConfigSoftMax(std::string name, std::string input, std::string type, int numClasses, double weightDecay)
	{
		m_numClasses = numClasses;
		m_weightDecay = weightDecay;
		m_name = name;
		m_input = input;
		m_type = type;
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
		m_fc.clear();
		m_conv.clear();
		m_softMax.clear();

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

	int getNonLinearity(){
		return m_nonLinearity->getValue();}

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

	const std::vector<ConfigConv*>& getConv(){
		return m_conv;
	}

	const std::vector<ConfigPooling*>& getPooling(){
		return m_pooling;
	}

	const std::vector<ConfigFC*>& getFC(){
		return m_fc;
	}

	const std::vector<ConfigSoftMax*>& getSoftMax(){
		return m_softMax;
	}

	const std::vector<ConfigBase*> getFirstLayers(){
		return m_firstLayers;
	}

	ConfigBase* getLayerByName(std::string name){
		if(m_layerMaps.find(name) != m_layerMaps.end()){
					return m_layerMaps[name];
		}
		else 
		{
			printf("layer %s does not exit\n", name.c_str());
			exit(0);
		}
	}

	void insertLayerByName(std::string name, ConfigBase* layer){
		if(m_layerMaps.find(name) == m_layerMaps.end()){
			m_layerMaps[name] = layer;
		}
		else {
			printf("layer %s exit\n", name.c_str());
			exit(0);
		}
	}

	void setImageSize(int imageSize){
		m_imageSize = imageSize;
	}

	int getImageSize(){
		return m_imageSize;
	}

private:
	void deleteComment();
	void deleteSpace();
	bool get_word_bool(std::string &str, std::string name);
	std::string get_word_type(std::string &str, std::string name);
	double get_word_double(std::string &str, std::string name);
	int get_word_int(std::string &str, std::string name);
	std::string read_2_string(std::string File_name);
	void get_layers_config(std::string &str);

	void init(std::string path);
	std::string m_configStr;
	std::string m_path;
	std::vector<ConfigFC*>m_fc;
	std::vector<ConfigConv*>m_conv;
	std::vector<ConfigPooling*>m_pooling;
	std::vector<ConfigSoftMax*>m_softMax;

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

	double momentum;
	double lrate;
	int m_imageSize;
};

#endif
