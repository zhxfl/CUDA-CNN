#ifndef __LAYERS_BASE_CU_H__
#define __LAYERS_BASE_CU_H__

class LayerBase
{
public:
	virtual void feedforward() = 0;
	virtual void backpropagation() = 0;
};

#endif
