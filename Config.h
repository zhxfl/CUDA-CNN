#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <string>
#include <vector>
#include "util.h"

class ConfigBase
{
};


class ConfigGradient:private ConfigBase
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


class ConfigBatchSize:private ConfigBase
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

class ConfigNonLinearity:private ConfigBase
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


class ConfigConv:private ConfigBase
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


class ConfigFC:private ConfigBase
{
public:
	ConfigFC(int numHiddenNeurons, double weightDecay,
		double dropoutRate)
	{
		m_numHiddenNeurons = numHiddenNeurons;
		m_weightDecay = weightDecay;
		m_dropoutRate = dropoutRate;
	}
	int m_numHiddenNeurons;
	double m_weightDecay;
	double m_dropoutRate;
};


class ConfigSoftMax:private ConfigBase
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

class Config
{
public:
	static Config* instance(){
		static Config *config = new Config("Config.txt");
		return config;
	}

	int getNonLinearity(){
		return m_nonLinearity->getValue();}

	bool getIsGradientChecking(){
		return m_isGrandientChecking->getValue();}

	int getBatchSize(){
		return m_batchSize->getValue();}

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
	Config(std::string path):m_path(path){
		init();
	}
	void deleteComment();
	void deleteSpace();
	bool get_word_bool(std::string &str, std::string name);
	std::string get_word_type(std::string &str, std::string name);
	double get_word_double(std::string &str, std::string name);
	int get_word_int(std::string &str, std::string name);
	std::string read_2_string(std::string File_name);
	void get_layers_config(std::string &str);

	void init();
	std::string m_configStr;
	std::string m_path;
	std::vector<ConfigFC*>m_fc;
	std::vector<ConfigConv*>m_conv;
	std::vector<ConfigSoftMax*>m_softMax;

	ConfigNonLinearity*m_nonLinearity;
	ConfigGradient* m_isGrandientChecking;
	ConfigBatchSize*m_batchSize;
};
#endif