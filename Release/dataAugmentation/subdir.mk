################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../dataAugmentation/cuTrasformation.cu \
../dataAugmentation/dataPretreatment.cu 

CU_DEPS += \
./dataAugmentation/cuTrasformation.d \
./dataAugmentation/dataPretreatment.d 

OBJS += \
./dataAugmentation/cuTrasformation.o \
./dataAugmentation/dataPretreatment.o 


# Each subdirectory must supply rules for building sources it contributes
dataAugmentation/%.o: ../dataAugmentation/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda-6.5/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 -gencode arch=compute_35,code=sm_35  -odir "dataAugmentation" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/cuda-6.5/NVIDIA_CUDA-6.5_Samples/common/inc/ -O3 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


