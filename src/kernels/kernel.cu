
#include "kernel.cuh"

#include <cmath> //for fmax, fmin functions

__global__ void test_kernel() {

	//thread mapping to an index
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	printf("test kernel worked! thread i:%d\n",i);
}


void test_kernel_wrapper(void){

// your code for initialization, copying data to device memory,
test_kernel<<<32,32>>>(); //kernel call
//your code for copying back the result to host memory & return
return;

}

__global__ void bgr_to_hsv_kernel_v1(unsigned char* bgrImage, unsigned char* hsvImage, int width, int height, int imageChannels)
{
	//determine column of single pixel in image based on thread index
	int col = threadIdx.x + blockDim.x*blockIdx.x; //x for column

	//determine row of single pixel in image based on thread index
	int row = threadIdx.y + blockDim.y*blockIdx.y; //y for row

	//prevent column and row from crossing boundaries
	if(col < width && row < height){
	 	
		int addr = row*width + col; //position of pixel in rgb image
		int blue_addr = imageChannels*addr; //position of red value of pixel in rgb image in relation to position of gray pixel

		float b = (float)bgrImage[blue_addr]; //blue component of rgb image pixel
		float g = (float)bgrImage[blue_addr + 1]; //green component of rgb image pixel
		float r = (float)bgrImage[blue_addr + 2]; //red component of rgb image pixel
		
		float h, s, v;
	
		r = r / 255.0f;
		g = g / 255.0f;
		b = b / 255.0f;
		
	
		float max = fmax(r, fmax(g, b));
		float min = fmin(r, fmin(g, b));
		float diff = max - min;
	
		v = max;
	
		if(v == 0.0f) { // black
			h = s = 0.0f;
		} else {
			s = diff / v;
			if(diff < 0.001f) { // grey
				h = 0.0f;
			} else { // color
				if(max == r) {
					h = 60.0f * (g - b)/diff;
					if(h < 0.0f) { h += 360.0f; }
				} else if(max == g) {
					h = 60.0f * (2 + (b - r)/diff);
				} else {
					h = 60.0f * (4 + (r - g)/diff);
				}
			}		
		}
	
		//assign the sum of the contributions of red, green, and blue components of rgb image pixel with weights to gray color value of gray pixel
		hsvImage[blue_addr] = (unsigned char)h;
		hsvImage[blue_addr + 1] = (unsigned char)s;
		hsvImage[blue_addr + 2] = (unsigned char)v;

	}
	
}

__global__ void BGR2YCrCb_kernel_naive(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels) {
	//************************************************************************************************
	//**
	//** convertBGR2YCrCb_Naive: convert BGR image kernel into YCrCb format.
	//**
	//** Methodology: Each thread converts three image channels for a pixel.
	//**
	//** Usage notes: Image kernels are stored in integer format and operated on using floating point operations
	//**              **width and height parameters are pixels, not bytes**
	//**
	//************************************************************************************************
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	float fY, fCr, fCb, fR, fG, fB;
	//data boundary check
	if (x < width && y < height) {
		// determine pixel channel offsets
		int ch0Offset = (y*width + x)*channels;
		int ch1Offset = (y*width + x)*channels + 1;
		int ch2Offset = (y*width + x)*channels + 2;
		fR  = inputImage[ch0Offset]; //get input pixel channel values
		fG  = inputImage[ch1Offset];
		fB  = inputImage[ch2Offset];
		fY  =  0.257 * fR + 0.504 * fG + 0.098 * fB +  16.0; //convert to YCrCb
		fCr = -0.148 * fR - 0.291 * fG + 0.439 * fB + 128.0;
		fCb =  0.439 * fR - 0.368 * fG - 0.071 * fB + 128.0;
		outputImage[ch0Offset] = (unsigned char)fY; //cast float YCrCb values to ints and store in output image array
		outputImage[ch1Offset] = (unsigned char)fCr;
		outputImage[ch2Offset] = (unsigned char)fCb;
	}
}

__global__ void YCrCb2BGR_kernel_naive(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels) {
	//************************************************************************************************
	//**
	//** convertYCrCb2BGR_Naive: convert YCrCb image kernel into BGR format.
	//**
	//** Methodology: Each thread converts three image channels for a pixel.
	//**
	//** Usage notes: Image kernels are stored in integer format and operated on using floating point operations
	//**              **width and height parameters are pixels, not bytes**
	//**
	//************************************************************************************************
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	float fY, fCr, fCb, fR, fG, fB;
	//data boundary check
	if (x < width && y < height) {
		// determine pixel channel offsets
		int ch0Offset = (y*width + x)*channels;
		int ch1Offset = (y*width + x)*channels + 1;
		int ch2Offset = (y*width + x)*channels + 2;
		fY  = (float)inputImage[ch0Offset]; //get input pixel channel values
		fCr = (float)inputImage[ch1Offset];
		fCb = (float)inputImage[ch2Offset];
		fG = std::fmin(((1.164 * (fY - 16.0)) - (0.813 * (fCr - 128.0)) - (0.392 * (fCb - 128.0))), 255.0); //convert to BGR
		fR = std::fmin( (1.164 * (fY - 16.0)) + (1.596 * (fCr - 128.0)), 255.0);
		fB = std::fmin( (1.164 * (fY - 16.0)) + (2.017 * (fCb - 128.0)), 255.0);
		outputImage[ch0Offset] = (unsigned char)fB; //cast float BGR values to ints and store in output array
		outputImage[ch1Offset] = (unsigned char)fG;
		outputImage[ch2Offset] = (unsigned char)fR;
	}
}

__global__ void BGR2YCrCb_kernel_optimized(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels) {
	//************************************************************************************************
	//**
	//** convertBGR2YCrCb_Optimized: convert BGR image kernel into YCrCb format.
	//**
	//** Methodology: Each thread converts three image channels for pixels in a strided fashion with loop unrolling.
	//**              For the ASML application, it is highly likely any dermoscopic image to be processes exceeds
	//**              ~350,000 pixels in size, so loop unrolling will take this into consideration
	//**
	//** Usage notes: Image kernels are stored in integer format and operated on using floating point operations
	//**              **width and height parameters are pixels, not bytes**
	//**
	//************************************************************************************************
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	float fY, fCr, fCb, fR, fG, fB;
	const int xStride = blockDim.x * gridDim.x;
	const int yStride = blockDim.y * gridDim.y;
	//data boundary check
	#pragma unroll
	while (x < width && y < height) {
		// determine pixel channel offsets
		int ch0Offset = (y*width + x)*channels;
		int ch1Offset = (y*width + x)*channels + 1;
		int ch2Offset = (y*width + x)*channels + 2;
		fR  = (float)inputImage[ch0Offset]; //get input pixel channel values
		fG  = (float)inputImage[ch1Offset];
		fB  = (float)inputImage[ch2Offset];
		fY  =  0.257 * fR + 0.504 * fG + 0.098 * fB +  16.0; //convert to YCrCb
		fCr = -0.148 * fR - 0.291 * fG + 0.439 * fB + 128.0;
		fCb =  0.439 * fR - 0.368 * fG - 0.071 * fB + 128.0;
		outputImage[ch0Offset] = (unsigned char)fY; //cast float YCrCb values to ints and store in output image array
		outputImage[ch1Offset] = (unsigned char)fCr;
		outputImage[ch2Offset] = (unsigned char)fCb;
		x+=xStride;
		y+=yStride;
	}
}

__global__ void YCrCb2BGR_kernel_optimized(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels) {
	//************************************************************************************************
	//**
	//** convertYCrCb2BGR_Optimized: convert YCrCb image kernel into BGR format.
	//**
	//** Methodology: Each thread converts three image channels for pixels in a strided fashion with loop unrolling.
	//**
	//** Usage notes: Image kernels are stored in integer format and operated on using floating point operations
	//**              **width and height parameters are pixels, not bytes**
	//**
	//************************************************************************************************
	int x = threadIdx.x+ blockIdx.x* blockDim.x;
	int y = threadIdx.y+ blockIdx.y* blockDim.y;
	float fY, fCr, fCb, fR, fG, fB;
	const int xStride = blockDim.x * gridDim.x;
	const int yStride = blockDim.y * gridDim.y;
	//data boundary check
	#pragma unroll
	while (x < width && y < height) {
		// determine pixel channel offsets
		int ch0Offset = (y*width + x)*channels;
		int ch1Offset = (y*width + x)*channels + 1;
		int ch2Offset = (y*width + x)*channels + 2;
		fY  = (float)inputImage[ch0Offset]; //get input pixel channel values
		fCr = (float)inputImage[ch1Offset];
		fCb = (float)inputImage[ch2Offset];
		fG = std::fmin(((1.164 * (fY - 16.0)) - (0.813 * (fCr - 128.0)) - (0.392 * (fCb - 128.0))), 255.0); //convert to BGR
		fR = std::fmin( (1.164 * (fY - 16.0)) + (1.596 * (fCr - 128.0)), 255.0);
		fB = std::fmin( (1.164 * (fY - 16.0)) + (2.017 * (fCb - 128.0)), 255.0);
		outputImage[ch0Offset] = (unsigned char)fB; //cast float BGR values to ints and store in output array
		outputImage[ch1Offset] = (unsigned char)fG;
		outputImage[ch2Offset] = (unsigned char)fR;
		x+=xStride;
		y+=yStride;
	}
}

	
void bgr_to_hsv_kernel_v1_wrapper(unsigned char* bgrImage, unsigned char* hsvImage, int width, int height, int imageChannels)
{
	//allocate memory to device
	unsigned char *deviceBGRImageData;
	unsigned char *deviceHSVImageData;
	
	cudaMalloc((void **)&deviceBGRImageData,
             width * height * imageChannels * sizeof(unsigned char));
	
	cudaMalloc((void **)&deviceHSVImageData,
             width * height * imageChannels * sizeof(unsigned char));
   
	//copy host input bgr image data to device
    cudaMemcpy(deviceBGRImageData, bgrImage,
             width * height * imageChannels * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
             
	//use 16x16 2d thread configuration in thread block
	dim3 blockDimensions(16,16,1); //256 threads = 16x16

	//set dimensions of grid for processing image pixels based on dividing image dimensions by single thread block dimensions.
	int xdim = (int)( ( width / 16) + 1);
	int ydim = (int)( ( height / 16) + 1);
	dim3 gridDimensions( xdim, ydim, 1);
	
	//call kernel function to convert bgr to hsv
	bgr_to_hsv_kernel_v1 <<<gridDimensions,blockDimensions>>>(deviceBGRImageData,deviceHSVImageData,width,height,imageChannels);
	
	cudaDeviceSynchronize();//wait for all threads to complete

	//copy device hsv image data to output
	cudaMemcpy(deviceHSVImageData, hsvImage,
             width * height * imageChannels * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);
	
	cudaFree(deviceHSVImageData);
	cudaFree(deviceBGRImageData);
}

void BGR2YCrCb_kernel_wrapper(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int imageChannels){
	//**
	//** Purpose:		convert BGR image to YCrCb format
	//**
	//** Parameters:	inputImage: the input image as an array of unsigned characters
	//**                outputImage: the output image as an array of unsigned characters
	//**				width: the width of the image in pixels
	//**				height: the height of the image in pixels
	//**                imageChannels the number of channels per pixel
	//**

	const int imgBytes  = width * height * imageChannels * sizeof(unsigned char); //calculate how many bytes are in the iput image
	unsigned char *deviceInputImageData;
	unsigned char *deviceOutputImageData;
	const unsigned int blockGridDivider = 1;

	const dim3 blockDim(32,32);
	// in the naive version each thread computes a single pixel
	const dim3 gridDim(((width-1)/(blockDim.x/blockGridDivider) + 1), ((height-1)/(blockDim.y/blockGridDivider) + 1));

	cudaMalloc((void **)&deviceInputImageData, imgBytes);  //allocate memory on the device
	cudaMalloc((void **)&deviceOutputImageData, imgBytes);  //allocate memory on the device
	cudaMemcpy(deviceInputImageData, inputImage, imgBytes, cudaMemcpyHostToDevice); //copy image array to device

	//execute the colorspace conversion
//	BGR2YCrCb_kernel_naive<<<gridDim, blockDim>>>(deviceInputImageData, deviceOutputImageData, width, height, imageChannels);
	BGR2YCrCb_kernel_optimized<<<gridDim, blockDim>>>(deviceInputImageData, deviceOutputImageData, width, height, imageChannels);

	cudaDeviceSynchronize();//wait for all threads to complete

	cudaMemcpy(outputImage, deviceOutputImageData, imgBytes, cudaMemcpyDeviceToHost); //copy converted image from

	cudaFree(deviceInputImageData); //free up device memory allocation
	cudaFree(deviceOutputImageData);//free up device memory allocation

	return;
}

void YCrCb2BGR_kernel_wrapper(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int imageChannels){
	//**
	//** Purpose:		convert BGR image to YCrCb format
	//**
	//** Parameters:	inputImage: the input image as an array of unsigned characters
	//**                outputImage: the output image as an array of unsigned characters
	//**				width: the width of the image in pixels
	//**				height: the height of the image in pixels
	//**                imageChannels the number of channels per pixel
	//**

	const int imgBytes  = width * height * imageChannels * sizeof(unsigned char); //calculate how many bytes are in the iput image
	unsigned char *deviceInputImageData;
	unsigned char *deviceOutputImageData;
	const unsigned int blockGridDivider = 4;

	const dim3 blockDim(32,32);
	// in the naive version each thread computes a single pixel
	const dim3 gridDim(((width-1)/(blockDim.x/blockGridDivider) + 1), ((height-1)/(blockDim.y/blockGridDivider) + 1));

	cudaMalloc((void **)&deviceInputImageData, imgBytes);  //allocate memory on the device
	cudaMalloc((void **)&deviceOutputImageData, imgBytes);  //allocate memory on the device
	cudaMemcpy(deviceInputImageData, inputImage, imgBytes, cudaMemcpyHostToDevice); //copy image array to device

	//execute the colorspace conversion
//	YCrCb2BGR_kernel_naive<<<gridDim, blockDim>>>(deviceInputImageData, deviceOutputImageData, width, height, imageChannels);
	YCrCb2BGR_kernel_optimized<<<gridDim, blockDim>>>(deviceInputImageData, deviceOutputImageData, width, height, imageChannels);

	cudaDeviceSynchronize();//wait for all threads to complete

	cudaMemcpy(outputImage, deviceOutputImageData, imgBytes, cudaMemcpyDeviceToHost); //copy converted image from

	cudaFree(deviceInputImageData); //free up device memory allocation
	cudaFree(deviceOutputImageData);//free up device memory allocation

	return;
}
