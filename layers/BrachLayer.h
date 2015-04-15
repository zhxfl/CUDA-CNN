/*
ref : ImageNet Classification with Deep Convolutional Neural Networks
*/
#ifndef __LAYERS_BRAHCH_LAYER_H__
#define __LAYERS_BRAHCH_LAYER_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include <map>
#include "../common/util.h"


class BrachLayer: public ConvLayerBase{
public:
	BrachLayer(std::string name);

	void feedforward();
	void backpropagation();
	void getGrad(){};
	void updateWeight(){};
	void clearMomentum(){};

	void calCost(){};
	void initFromCheckpoint(FILE* file){};
	void save(FILE* file){};

	~BrachLayer(){
		for(int i = 0; i < outputs.size(); i++){
			delete outputs[i];
		}
		for(int i = 0; i < curDelta.size(); i++){
			delete curDelta[i];
		}
	}

	cuMatrix<double>* getOutputs(std::string name){
		if(mapId.find(name) != mapId.end()){
			return outputs[mapId[name]];
		}else{
			printf("brachLayer: can not find getOutputs %s\n", name.c_str());
			exit(0);
			return NULL;
		}
	};

	cuMatrix<double>* getCurDelta(std::string name){
		if(mapId.find(name) != mapId.end()){
			return curDelta[mapId[name]];
		}else{
			printf("brachLayer: can not find getCurDelta %s\n", name.c_str());
			exit(0);
			return NULL;
		}
	}

	void setPreDelta(cuMatrix<double>* _preDelta){
		preDelta = _preDelta;
	}

	int getOutputAmount(){
		return outputAmount;
	}

	int getOutputDim(){
		return outputDim;
	}

	void printParameter(){};

private:
	cuMatrix<double>* inputs;
	cuMatrix<double>* preDelta;


	std::vector<cuMatrix<double>*> outputs;
	std::vector<cuMatrix<double>*> curDelta;
	
	std::map<std::string, int>mapId;

	int batch;
};
#endif
