#ifndef _CU_DISTORTION_H_
#define _CU_DISTORTION_H_

void cuInitDistortionMemery(int batch, int ImgSize);
void cuApplyRandom(int batch, unsigned long long s, double dElasticScaling, int ImgSize);
void cuApplyDistortion(double**inputs, double**outputs, int batch, int ImgSize);
void cuApplyCrop(double**inputs, double**outputs, int batch,int ImgSize);

#endif _DISTORTION_H_