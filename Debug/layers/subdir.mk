################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../layers/LayerBase.cpp 

CU_SRCS += \
../layers/Pooling.cu 

CU_DEPS += \
./layers/Pooling.d 

OBJS += \
./layers/LayerBase.o \
./layers/Pooling.o 

CPP_DEPS += \
./layers/LayerBase.d 


# Each subdirectory must supply rules for building sources it contributes
layers/%.o: ../layers/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/home/zhxfl/NVIDIA_CUDA-6.5_Samples/common/inc/ -G -g -O0 -gencode arch=compute_50,code=sm_50  -odir "layers" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/home/zhxfl/NVIDIA_CUDA-6.5_Samples/common/inc/ -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

layers/%.o: ../layers/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/home/zhxfl/NVIDIA_CUDA-6.5_Samples/common/inc/ -G -g -O0 -gencode arch=compute_50,code=sm_50  -odir "layers" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/home/zhxfl/NVIDIA_CUDA-6.5_Samples/common/inc/ -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


