#ifndef __LAYERS_BASE_CU_H__
#define __LAYERS_BASE_CU_H__

#include "../common/cuMatrix.h"
#include "../common/cuMatrixVector.h"


class LayerBase
{
public:
	virtual void feedforward() = 0;
	virtual void backpropagation() = 0;
	virtual void getGrad() = 0;
	virtual void updateWeight() = 0;
	virtual void clearMomentum() = 0;
	virtual void getCost(cuMatrix<double>*cost, int* y = NULL) = 0;
	virtual void save(FILE* file) = 0;
	virtual void initFromCheckpoint(FILE* file) = 0;


	virtual cuMatrix<double>* getOutputs() = 0;
	virtual cuMatrix<double>* getPreDelta() = 0;
	virtual cuMatrix<double>* getCurDelta() = 0;

	virtual void printParameter() = 0;

	std::string m_name;
	std::vector<std::string> m_preLayer;
};

class ConvLayerBase: public LayerBase
{
public:
	int inputDim ;
	int outputDim;
	int inputAmount;
	int amount;
	int outputAmount;
};

class Layers
{
public:
	static Layers* instance(){
		static Layers* layers= new Layers();
		return layers;
	}
	LayerBase* get(std::string name);
	void set(std::string name, LayerBase* layer);

	void setInputs(cuMatrixVector<double>*m_inputs);
	cuMatrixVector<double>* getInputs();

private:
	std::map<std::string, LayerBase*>m_maps;
	cuMatrixVector<double>*m_inputs;
};
#endif
