################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Config.cpp \
../cuMatrix.cpp \
../main.cpp \
../readMnistData.cpp \
../util.cpp 

CU_SRCS += \
../cuDistortion.cu \
../net.cu 

CU_DEPS += \
./cuDistortion.d \
./net.d 

OBJS += \
./Config.o \
./cuDistortion.o \
./cuMatrix.o \
./main.o \
./net.o \
./readMnistData.o \
./util.o 

CPP_DEPS += \
./Config.d \
./cuMatrix.d \
./main.d \
./readMnistData.d \
./util.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.0/bin/nvcc -O3 -gencode arch=compute_20,code=sm_20  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.0/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.0/bin/nvcc -O3 -gencode arch=compute_20,code=sm_20  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.0/bin/nvcc --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


