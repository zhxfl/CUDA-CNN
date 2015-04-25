>CUDA-CNN
>========


>Document   
>1.  The simple c version author is <a href="http://eric-yuan.me/cnn/"> Eric </a>   
>2.  <a href="http://zhxfl.github.io/cuda-cnn_cuda-stream"> Overlap Data Transfers in CUDA </a>   
>
>Results
>--------
>CNN accelerated by cuda.   
>The <a href="http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html"> start-of-art result's</a> of popular datasets    
>1. Test on <a href="http://yann.lecun.com/exdb/mnist/"> mnist</a> and get 99.76%, after voting(99.82%) (best 99.79%)   
>2. Test on cifar-10  and get 85.08%   (best 89%)   
***

>Feature
>--------
>1. Use ***<a href="http://cs.nyu.edu/~wanli/dropc/">Dropout</a>*** to train the NetWork
>2. Support checkpoint, the program will save the best test result and save the network weight in the file "Result/checkPoint.txt", If the program exit accidentally, you can continue the program form this checkpoint.
>3. Translate the data set of mnist, including scale, rotate, ***distortion***,
 accordding to <a href="http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=D1C7D701BD39935473808DA5A93426C5?doi=10.1.1.160.8494&rep=rep1&type=pdf">Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis</a>.
>4. The log will be saved in the file "Result/log.txt".  
>5. In the convolutional layers, you can chose ***combine feature maps***, according to <a href="http://cogprints.org/5869/1/cnn_tutorial.pdf">notes on Convolutional Neural NetWorks</a>.      
>6. Support <a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks">local connection layers</a>.   
>7. If you want the program run fast, you can set the "TEST_EPOCH" to be large.     
>8. Support ***branchLayer*** and ***combineLayer***, which is designed accordding to ***<a href="http://arxiv.org/abs/1409.4842">goolenet</a>***, the network structure is no logger an linear structure but Directed acycline graph.
***

>Compile
>-------
>Depend on opencv and cuda    
>You can compile the code on windows or linux.   
###SDK include path(-I)   
>* linux: /usr/local/cuda/samples/common/inc/ (For include file "helper_cuda"); /usr/local/include/opencv/ (Depend on situation)        
>* windows: X:/Program Files (x86) /NVIDIA Corporation/CUDA Samples/v6.5/common/inc (For include file "helper_cuda"); X:/Program Files/opencv/vs2010/install/include (Depend on situation)
>
###Library search path(-L)   
>* linux: /usr/local/lib/   
>* windows: X:/Program Files/opencv/vs2010/install/x86/cv10/lib (Depend on situation)    
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


###Windows
>1. Install vs2010.
>2. Download and install <a href="http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0-beta/"> opencv-2.4</a> or other higher versions
>3. Download and install <a href="https://developer.nvidia.com/cuda-downloads"> cuda-5.0</a> or other higher versions
>4. When you create a new project using VS2010, You can find NVIDIA-CUDA project template, create a cuda-project.
>5. View-> Property Pages-> Configuration Properties-> CUDA C/C++ -> Device-> Code Generation-> compute_20,sm_20   
>6. View-> Property Pages-> Configuration Properties-> CUDA C/C++ -> Common-> Generate Relocatable Device Code-> Yes(-rdc=true) 
>7. View-> Property Pages-> Configuration Properties-> Linker-> Input-> Additional Dependencies-> libraries(-l)   
>8. View-> Property Pages-> Configuration Properties-> VC++ Directories-> General-> Library search path(-L)  
>9. View-> Property Pages-> Configuration Properties-> VC++ Directories-> General-> Include Directories(-I)  

###Linux
>1. Install opencv and cuda
>2. Start the ***nsight*** from cuda
>3. Create an 'empty cuda' project and import the clone code  
>4. Project->Proerties for add-> Build-> Settings->CUDA->Device linker mode: separate compilation   
>5. Project->Proerties for add-> Build-> Settings->CUDA->Generate PTX code 2.0
>6. Project->Proerties for add-> Build-> Settings->CUDA->Generate GPU code 2.0
>7. Project->Proerties for add-> Build-> Settings->Tool Settings->NVCC Compiler->includes: +/usr/local/cuda/samples/common/inc/; + opencv sdk include path ;   
>8. Project->Proerties for add-> Build-> Settings->Tool Settings->NVCC Linkers->Libraries: libraries(-l)   
>9. Project->Proerties for add-> Build-> Settings->Tool Settings->NVCC Linkers->Libraries search path(-L): /usr/local/lib/    

***
>Config   
>1. <a href="https://github.com/zhxfl/CUDA-CNN/blob/master/Config/Cifar10Config.txt">CIFAR10</a>   
>2. <a href="https://github.com/zhxfl/CUDA-CNN/blob/master/Config/MnistConfig.txt">MNIST</a>   
***

>Informations
>------------
>* Author :zhxfl  
>* Mail   :zhxfl@mail.ustc.edu.cn  
>* Welcome for any suggest!!   
>* 

