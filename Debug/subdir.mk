################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Config.cpp \
../MemoryMonitor.cpp \
../cuMatrix.cpp \
../cuMatrixVector.cpp \
../main.cpp \
../util.cpp 

CU_SRCS += \
../cuTrasformation.cu \
../dataPretreatment.cu \
../net.cu 

CU_DEPS += \
./cuTrasformation.d \
./dataPretreatment.d \
./net.d 

OBJS += \
./Config.o \
./MemoryMonitor.o \
./cuMatrix.o \
./cuMatrixVector.o \
./cuTrasformation.o \
./dataPretreatment.o \
./main.o \
./net.o \
./util.o 

CPP_DEPS += \
./Config.d \
./MemoryMonitor.d \
./cuMatrix.d \
./cuMatrixVector.d \
./main.d \
./util.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/home/zhxfl/NVIDIA_CUDA-6.5_Samples/common/inc/ -G -g -O0 -gencode arch=compute_50,code=sm_50  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/home/zhxfl/NVIDIA_CUDA-6.5_Samples/common/inc/ -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/home/zhxfl/NVIDIA_CUDA-6.5_Samples/common/inc/ -G -g -O0 -gencode arch=compute_50,code=sm_50  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/home/zhxfl/NVIDIA_CUDA-6.5_Samples/common/inc/ -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


