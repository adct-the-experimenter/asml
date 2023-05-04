#ifndef KERNELS_CUH__
#define KERNELS_CUH__

#include <stdio.h>
#include "../implementation_version.h"

void test_kernel_wrapper();

void bgr_to_hsv_kernel_v1_wrapper(unsigned char* bgrImage, unsigned char* hsvImage, int width, int height, int imageChannels);

void BGR2YCrCb_kernel_wrapper(unsigned char* bgrImage, unsigned char* YCrCbImage, int width, int height, int imageChannels);
void YCrCb2BGR_kernel_wrapper(unsigned char* bgrImage, unsigned char* YCrCbImage, int width, int height, int imageChannels);

void BGR2GRAY_wrapper(float *hostOutputArray, float *hostInputArray, int width, int height);
void RGB2GRAY_wrapper(float *hostOutputArray, float *hostInputArray, int width, int height);
#endif
