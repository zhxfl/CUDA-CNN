#include "LayerBase.h"
#include "../common/cuMatrixVector.h"

LayerBase* Layers::get(std::string name)
{
	if(m_maps.find(name) != m_maps.end()){
		return m_maps[name];
	}else{
		printf("layer:get %s not exit\n", name.c_str());
		exit(0);
		return NULL;
	}
}

void Layers::set(std::string name, LayerBase* layer)
{
	if(m_maps.find(name) == m_maps.end()){
		m_maps[name] = layer;
	}
	else {
		printf("layers:set %s exit\n", name.c_str());
	}
}