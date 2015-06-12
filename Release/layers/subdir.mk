################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../layers/LayerBase.cpp 

CU_SRCS += \
../layers/BranchLayer.cu \
../layers/CombineLayer.cu \
../layers/ConvCFM.cu \
../layers/DataLayer.cu \
../layers/FullConnect.cu \
../layers/LRN.cu \
../layers/LocalConnect.cu \
../layers/NIN.cu \
../layers/One.cu \
../layers/Pooling.cu \
../layers/SoftMax.cu 

CU_DEPS += \
./layers/BranchLayer.d \
./layers/CombineLayer.d \
./layers/ConvCFM.d \
./layers/DataLayer.d \
./layers/FullConnect.d \
./layers/LRN.d \
./layers/LocalConnect.d \
./layers/NIN.d \
./layers/One.d \
./layers/Pooling.d \
./layers/SoftMax.d 

OBJS += \
./layers/BranchLayer.o \
./layers/CombineLayer.o \
./layers/ConvCFM.o \
./layers/DataLayer.o \
./layers/FullConnect.o \
./layers/LRN.o \
./layers/LayerBase.o \
./layers/LocalConnect.o \
./layers/NIN.o \
./layers/One.o \
./layers/Pooling.o \
./layers/SoftMax.o 

CPP_DEPS += \
./layers/LayerBase.d 


# Each subdirectory must supply rules for building sources it contributes
layers/%.o: ../layers/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda-6.5/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 -gencode arch=compute_50,code=sm_50  -odir "layers" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda-6.5/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 --compile --relocatable-device-code=true -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

layers/%.o: ../layers/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda-6.5/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 -gencode arch=compute_50,code=sm_50  -odir "layers" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda-6.5/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


