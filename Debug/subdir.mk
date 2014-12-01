################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../main.cpp \
../readData.cpp \
../util.cpp 

CU_SRCS += \
../cuDistortion.cu \
../cuMatrix.cu \
../cuPoints.cu \
../net.cu 

CU_DEPS += \
./cuDistortion.d \
./cuMatrix.d \
./cuPoints.d \
./net.d 

OBJS += \
./cuDistortion.o \
./cuMatrix.o \
./cuPoints.o \
./main.o \
./net.o \
./readData.o \
./util.o 

CPP_DEPS += \
./main.d \
./readData.d \
./util.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.0/bin/nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.0/bin/nvcc --compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.0/bin/nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20  -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.0/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


