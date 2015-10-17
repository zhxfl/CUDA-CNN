################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../dataAugmentation/cuTrasformation.cu \
../dataAugmentation/dataPretreatment.cu \
../dataAugmentation/pca.cu 

CU_DEPS += \
./dataAugmentation/cuTrasformation.d \
./dataAugmentation/dataPretreatment.d \
./dataAugmentation/pca.d 

OBJS += \
./dataAugmentation/cuTrasformation.o \
./dataAugmentation/dataPretreatment.o \
./dataAugmentation/pca.o 


# Each subdirectory must supply rules for building sources it contributes
dataAugmentation/%.o: ../dataAugmentation/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -I/usr/local/cuda-7.0/NVIDIA_CUDA-7.0_Samples/common/inc/ -O3 -gencode arch=compute_50,code=sm_50  -odir "dataAugmentation" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -I/usr/local/cuda-7.0/NVIDIA_CUDA-7.0_Samples/common/inc/ -O3 --compile --relocatable-device-code=true -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


