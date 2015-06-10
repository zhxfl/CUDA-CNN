#ifndef _CU_DISTORTION_H_
#define _CU_DISTORTION_H_

void cuInitDistortionMemery(int batch, int ImgSize);
void cuApplyRandom(int batch, unsigned long long s, int ImgSize);
void cuApplyDistortion(float**inputs, float**outputs, int batch, int ImgSize);
void cuApplyCropRandom(float**inputs, float**outputs, int batch, int ImgSize);
void cuApplyCrop(float**inputs, float**outputs, int batch, int ImgSize, int cropr, int cropc);
void cuApplyHorizontal(float **inputs, float**outputs, int batch, int ImgSize, int flag);
void cuApplyScaleAndRotate(int batch, int ImgSize, float scalingx, float scalingy, float rotation);
void cuApplyWhiteNoise(float **inputs, float**outputs, int batch, int ImgSize, float stdev);


#define RANDOM_HORIZONTAL 0
#define HORIZONTAL 1
#define NOT_HORIZONTAL 2

#endif
