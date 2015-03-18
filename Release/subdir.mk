################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../main.cpp 

CU_SRCS += \
../net.cu 

CU_DEPS += \
./net.d 

OBJS += \
./main.o \
./net.o 

CPP_DEPS += \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 -gencode arch=compute_50,code=sm_50  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 -gencode arch=compute_50,code=sm_50  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 --compile --relocatable-device-code=true -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


