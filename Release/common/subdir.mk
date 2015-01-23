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

OBJS += \
./common/Config.o \
./common/MemoryMonitor.o \
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
	/usr/local/cuda-6.5/bin/nvcc -I/home/zhxfl/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 -gencode arch=compute_50,code=sm_50  -odir "common" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/home/zhxfl/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


