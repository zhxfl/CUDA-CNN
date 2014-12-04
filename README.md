>CUDA-CNN
>========



>Functions
>--------
>CNN accelerated by cuda. Test on mnist and finilly get 99.7%
***


>Feature
>--------
>1. Use ***DropConnnect*** to train the NetWork
>2. Support checkpoint, the program will save the best test result and save the network weight in the file "net.txt"
>3. Translate the data set of mnist, including scale, rotate, ***distortion***.
>

***

>Compile
>-------
>Depend on opencv and cuda
>You can compile the code on windows or linux.
###Windows###
>1. Install vs2010.
>2. Download and install <a href="http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0-beta/" title="opencv-3.0"> opencv-3.0</a> 
>3. Download and install <a href="https://developer.nvidia.com/cuda-downloads", title="cuda-6.5"> cuda-6.5</a>
>4. When you create a new project using VS2010, You can find NVIDIA-CUDA-6.5 project template, create a cuda-project.
>5. Include 

