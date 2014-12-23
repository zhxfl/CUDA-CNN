#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <string>
#include <vector>
#include "util.h"

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

class ConfigCombineFeatureMaps
{
public:
	ConfigCombineFeatureMaps(bool combine){
		m_cfm = combine;
	}
	bool getValue(){return m_cfm;}
private:
	bool m_cfm;
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

class ConfigConv
{
public:
	ConfigConv(int kernelSize, int amount, double weightDecay, int poolingDim){
		m_kernelSize = kernelSize;
		m_amount = amount;
		m_weightDecay = weightDecay;
		m_poolingDim = poolingDim;
	}
	int m_kernelSize;
	int m_amount;
	double m_weightDecay;
	int m_poolingDim;
};

class ConfigFC
{
public:
	ConfigFC(int numFullConnectNeurons, double weightDecay,
		double dropoutRate)
	{
		m_numFullConnectNeurons = numFullConnectNeurons;
		m_weightDecay = weightDecay;
		m_dropoutRate = dropoutRate;
	}
	int m_numFullConnectNeurons;
	double m_weightDecay;
	double m_dropoutRate;
};

class ConfigSoftMax
{
public:
	int m_numClasses;
	double m_weightDecay;
	ConfigSoftMax(int numClasses, double weightDecay)
	{
		m_numClasses = numClasses;
		m_weightDecay = weightDecay;
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
	void initPath(std::string path){
		m_path = path;
		init(m_path);
	}
	static Config* instance(){
		static Config* config = new Config();
		return config;
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

	bool getCFM(){
		return m_cfm->getValue();
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

	const std::vector<ConfigFC*>& getFC(){
		return m_fc;
	}

	const std::vector<ConfigSoftMax*>& getSoftMax(){
		return m_softMax;
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
	std::vector<ConfigSoftMax*>m_softMax;

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
	ConfigCombineFeatureMaps *m_cfm;
};

#endif
