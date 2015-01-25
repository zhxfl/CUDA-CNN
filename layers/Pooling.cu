#include "Pooling.h"
#include <vector>

void Pooling::feedforward()
{
	printf("Pooling feedforword\n");
}

void Pooling::backpropagation()
{
	printf("Pooling backpropagation\n");
}

Pooling::Pooling(std::vector<cuMatrix<double>*>&_inputs)
{
/*	for(std::vector<cuMatrix<double>*>iterator it = _inputs.begin();
			it != _inputs.end();
			it++){
		inputs.push_back(it);
	}
	*/
}
