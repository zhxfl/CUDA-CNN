>CUDA-CNN
>========

>Functions
>--------
>CNN accelerated by cuda.    
>Test on mnist and get 99.7%       (best 99.74%)
>Test on cifar-10 and get 81.38%   (best 90%)
>Test on cifar-100 and get 51.13%   (best 65%)
***

>Feature
>--------
>1. Use ***DropConnnect*** to train the NetWork
>2. Support checkpoint, the program will save the best test result and save the network weight in the file "checkPoint.txt", If the program exit accidentally, you can continue the program form this checkpoint.
>3. Translate the data set of mnist, including scale, rotate, ***distortion***.
>4. The log will be saved in the file "log.txt".  
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
>* opencv_imgcodecs
>* ***cublas***   
>* ***curand***   
>
###GPU compute 
>* capability 2.0   
>
###Windows
>1. Install vs2010.
>2. Download and install <a href="http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0-beta/" title="opencv-2.4"> opencv-2.4</a> or other higher versions
>3. Download and install <a href="https://developer.nvidia.com/cuda-downloads", title="cuda-5.0"> cuda-5.0</a> or other higher versions
>4. When you create a new project using VS2010, You can find NVIDIA-CUDA project template, create a cuda-project.
>5. Add the "include path" and "lib path" to the project
>
###Linux
>1. Install opencv and cuda
>2. Start the ***nsight*** from cuda
>3. Create an 'empty cuda' project and import the clone code   
>4. Add the "include path" and "lib path" to the project
>

***

>Config
>-----------
###MNIST
>`#Comment#`   
>   
>`IS_GRADIENT_CHECKING = false;   #is true when debug#`   
>`BATCH_SIZE = 200;               #test image size should be divided with no remainder#`   
>`NON_LINEARITY = NL_RELU;        #NON_LINEARITY CAN = NL_SIGMOID , NL_TANH , NL_RELU#`   
>`CHANNELS = 1;                   #1, 3, 4#`   
>`CROP = 4;                       #0<= crop <=imgSize#`   
>`SCALE = 13.0;                   #13% of ImgSize#`   
>`ROTATION = 13.0;                #angle#`   
>`DISTORTION = 3.5;               #just for mnist#`   
>`SHOWIMAGE = false;              #show the images after transformation#`   
>`HORIZONTAL = false;             #horizontal reflection#`   
>`COMBINE_FEATRUE_MAPS = false;   #According to paper "notes on Convolutional Neural NetWorks"#`
>
###CIFAR-10
>
>`IS_GRADIENT_CHECKING = false;   #is true when debug#`   
>`BATCH_SIZE = 100;               #test image size should be divided with no remainder#`   
>`NON_LINEARITY = NL_RELU;        #NON_LINEARITY CAN = NL_SIGMOID , NL_TANH , NL_RELU#`   
>`CHANNELS = 3;                   #1, 3, 4#`   
>`CROP = 4;                       #0<= crop <=imgSize#`   
>`SCALE = 0.0;                    #13% of ImgSize#`   
>`ROTATION = 0.0;                 #angle#`   
>`DISTORTION = 0.0;               #just for mnist#`   
>`SHOWIMAGE = false;              #show the images after transformation#`   
>`HORIZONTAL = true;              #horizontal reflection#` 
>`COMBINE_FEATRUE_MAPS = false;   #According to paper "notes on Convolutional Neural NetWorks"#`
>
***

>Informations
>------------
>* Author :zhxfl  
>* Mail   :zhxfl@mail.ustc.edu.cn  
>* Welcome for any suggest!!   

