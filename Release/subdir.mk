################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Config.cpp \
../cuMatrix.cpp \
../cuMatrixVector.cpp \
../main.cpp \
../readCIFAR10Data.cpp \
../readMnistData.cpp \
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
./cuMatrix.o \
./cuMatrixVector.o \
./cuTrasformation.o \
./dataPretreatment.o \
./main.o \
./net.o \
./readCIFAR10Data.o \
./readMnistData.o \
./util.o 

CPP_DEPS += \
./Config.d \
./cuMatrix.d \
./cuMatrixVector.d \
./main.d \
./readCIFAR10Data.d \
./readMnistData.d \
./util.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.0/bin/nvcc -I/usr/local/cuda-6.0/samples/common/inc -O3 -gencode arch=compute_20,code=sm_20  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.0/bin/nvcc -I/usr/local/cuda-6.0/samples/common/inc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.0/bin/nvcc -I/usr/local/cuda-6.0/samples/common/inc -O3 -gencode arch=compute_20,code=sm_20  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.0/bin/nvcc --compile -I/usr/local/cuda-6.0/samples/common/inc -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


