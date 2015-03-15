################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../layers/LayerBase.cpp 

CU_SRCS += \
../layers/ConvCFM.cu \
../layers/ConvNCFM.cu \
../layers/FullConnnect.cu \
../layers/LocalConnect.cu \
../layers/Pooling.cu \
../layers/SoftMax.cu 

CU_DEPS += \
./layers/ConvCFM.d \
./layers/ConvNCFM.d \
./layers/FullConnnect.d \
./layers/LocalConnect.d \
./layers/Pooling.d \
./layers/SoftMax.d 

OBJS += \
./layers/ConvCFM.o \
./layers/ConvNCFM.o \
./layers/FullConnnect.o \
./layers/LayerBase.o \
./layers/LocalConnect.o \
./layers/Pooling.o \
./layers/SoftMax.o 

CPP_DEPS += \
./layers/LayerBase.d 


# Each subdirectory must supply rules for building sources it contributes
layers/%.o: ../layers/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda-6.5/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 -gencode arch=compute_35,code=sm_35  -odir "layers" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda-6.5/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

layers/%.o: ../layers/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda-6.5/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 -gencode arch=compute_35,code=sm_35  -odir "layers" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda-6.5/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


