################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../readData/readCIFAR100Data.cpp \
../readData/readCIFAR10Data.cpp \
../readData/readChineseData.cpp \
../readData/readMnistData.cpp 

OBJS += \
./readData/readCIFAR100Data.o \
./readData/readCIFAR10Data.o \
./readData/readChineseData.o \
./readData/readMnistData.o 

CPP_DEPS += \
./readData/readCIFAR100Data.d \
./readData/readCIFAR10Data.d \
./readData/readChineseData.d \
./readData/readMnistData.d 


# Each subdirectory must supply rules for building sources it contributes
readData/%.o: ../readData/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda-6.5/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 -gencode arch=compute_50,code=sm_50  -odir "readData" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda-6.5/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


