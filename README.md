>CUDA-CNN
>========


>Document   
>1.  <a href="http://zhxfl.github.io/cuda-cnn_cuda-stream"> Overlap Data Transfers in CUDA </a>
>
>Functions
>--------
>CNN accelerated by cuda.   
>The <a href="http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html"> start-of-art result's</a> of popular datasets    
>1. Test on <a href="http://yann.lecun.com/exdb/mnist/"> mnist</a> and get 99.76%, after voting(99.82%) (best 99.79%)   
>2. Test on cifar-10  and get 81.38%   (best 90%)   
>3. Test on cifar-100 and get 51.13%   (best 65%)   
***

>Feature
>--------
>1. Use ***DropConnnect*** to train the NetWork
>2. Support checkpoint, the program will save the best test result and save the network weight in the file "Result/checkPoint.txt", If the program exit accidentally, you can continue the program form this checkpoint.
>3. Translate the data set of mnist, including scale, rotate, ***distortion***.
>4. The log will be saved in the file "Result/log.txt".  
>5. In the convolutional layers, you can chose ***combine feature maps***, according to "notes on Convolutional Neural NetWorks"
>

***

>Compile
>-------
>Depend on opencv and cuda    
>You can compile the code on windows or linux.   
###SDK path   
>* linux: /usr/local/cuda/samples/common/inc/ (For include file "helper_cuda")      
>* windows: X:/Program Files (x86) /NVIDIA Corporation/CUDA Samples/v6.5/common/inc (For include file "helper_cuda")   
>
###Library search path(-L)   
>* linux: /usr/local/lib/   
>* windows: X：/Program Files/opencv/vs2010/install/x86/cv10/lib (Depend on situation)    
>
###libraries(-l)      
>* opencv_core   
>* opencv_highgui   
>* opencv_imgproc   
>* opencv_imgcodecs (need for opencv3.0)   
>* ***cublas***   
>* ***curand***   
>* ***cudadevrt***   
>
###GPU compute 
>* capability 2.0   
>* View -> Property Pages -> Configuration Properties -> CUDA C/C++ -> Device -> Code Generation -> compute_20,sm_20   
>* View -> Property Pages -> Configuration Properties -> CUDA C/C++ -> Common -> Generate Relocatable Device Code -> Yes (-rdc=true)   

###Windows
>1. Install vs2010.
>2. Download and install <a href="http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0-beta/"> opencv-2.4</a> or other higher versions
>3. Download and install <a href="https://developer.nvidia.com/cuda-downloads"> cuda-5.0</a> or other higher versions
>4. When you create a new project using VS2010, You can find NVIDIA-CUDA project template, create a cuda-project.
>5. View->Property Pages -> Configuration Properties -> Linker -> Input -> Additional Dependencies -> libraries(-l)   
>6. View->Property Pages -> VC++ Directories->General->Include Directories-> Library search path(-L)   

###Linux
>1. Install opencv and cuda
>2. Start the ***nsight*** from cuda
>3. Create an 'empty cuda' project and import the clone code  
>4. Project->Proerties->Settings->CUDA->Device linker mode: separate compilation   
>5. Project->Proerties->Settings->CUDA->Generate PTX code 3.5
>6. Project->Proerties->Settings->CUDA->Generate GPU code 3.5
>7. Project->Proerties->Settings->Tool Settings->NVCC Compiler->includes: +/usr/local/cuda/samples/common/inc/; +/usr/local/lib/ ;   
>8. Project->Proerties->Settings->Tool Settings->NVCC Linkers->Libraries: libraries(-l)    
>

***
>Config   
>1. <a href="https://github.com/zhxfl/CUDA-CNN/blob/master/Config/Cifar10Config.txt">CIFAR10</a>   
>2. <a href="https://github.com/zhxfl/CUDA-CNN/blob/master/Config/MnistConfig.txt">MNIST</a>   
>3. <a href="https://github.com/zhxfl/CUDA-CNN/blob/master/Config/Cifar100Config.txt">CIFAR100</a>
***

>Informations
>------------
>* Author :zhxfl  
>* Mail   :zhxfl@mail.ustc.edu.cn  
>* Welcome for any suggest!!   

