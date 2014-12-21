#ifndef _CU_DISTORTION_H_
#define _CU_DISTORTION_H_

void cuInitDistortionMemery(int batch, int ImgSize);
void cuApplyRandom(int batch, unsigned long long s, int ImgSize);
void cuApplyDistortion(double**inputs, double**outputs, int batch, int ImgSize);
void cuApplyCropRandom(double**inputs, double**outputs, int batch, int ImgSize);
void cuApplyCrop(double**inputs, double**outputs, int batch, int ImgSize, int cropr, int cropc);
void cuApplyHorizontal(double **inputs, double**outputs, int batch, int ImgSize);
void cuApplyScaleAndRotate(int batch, int ImgSize, double scaling, double rotation);

#endif
