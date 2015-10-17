################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../common/Config.cpp \
../common/MemoryMonitor.cpp \
../common/cuMatrix.cpp \
../common/cuMatrixVector.cpp \
../common/util.cpp 

CU_SRCS += \
../common/cuBase.cu 

CU_DEPS += \
./common/cuBase.d 

OBJS += \
./common/Config.o \
./common/MemoryMonitor.o \
./common/cuBase.o \
./common/cuMatrix.o \
./common/cuMatrixVector.o \
./common/util.o 

CPP_DEPS += \
./common/Config.d \
./common/MemoryMonitor.d \
./common/cuMatrix.d \
./common/cuMatrixVector.d \
./common/util.d 


# Each subdirectory must supply rules for building sources it contributes
common/%.o: ../common/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -I/usr/local/cuda-7.0/NVIDIA_CUDA-7.0_Samples/common/inc/ -O3 -gencode arch=compute_50,code=sm_50  -odir "common" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -I/usr/local/cuda-7.0/NVIDIA_CUDA-7.0_Samples/common/inc/ -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

common/%.o: ../common/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -I/usr/local/cuda-7.0/NVIDIA_CUDA-7.0_Samples/common/inc/ -O3 -gencode arch=compute_50,code=sm_50  -odir "common" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -I/usr/local/cuda-7.0/NVIDIA_CUDA-7.0_Samples/common/inc/ -O3 --compile --relocatable-device-code=true -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


