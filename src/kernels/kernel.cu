
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

__global__ void bgr_to_hsv_kernel(unsigned char* bgrImage, unsigned char* hsvImage, int width, int height, int imageChannels)
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
		//a = pixel.w;
	
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

	
void bgr_to_hsv_kernel_wrapper(unsigned char* bgrImage, unsigned char* hsvImage, int width, int height, int imageChannels)
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
	bgr_to_hsv_kernel <<<gridDimensions,blockDimensions>>>(deviceBGRImageData,deviceHSVImageData,width,height,imageChannels);
	
	//copy device hsv image data to output
	cudaMemcpy(deviceHSVImageData, hsvImage,
             width * height * imageChannels * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);
	
	cudaFree(deviceHSVImageData);
	cudaFree(deviceBGRImageData);
}


//Kevin BGR 2 GRAY

#define CHANNELS 3
__global__ void BGR2GRAY(float *outputArray, float *inputArray, int width, int height){
    //Getting thread indexies
    int c = threadIdx.x+blockIdx.x*blockDim.x;
    int r = threadIdx.y+blockIdx.y*blockDim.y;
    //Converting from BGR to GRAYSCALE
    if (c < width && r < height){
        int grayOffset = r*width + c;
        int rgbOffset = grayOffset*CHANNELS;
        float b = inputArray[rgbOffset];
        float g = inputArray[rgbOffset+1];
        float r = inputArray[rgbOffset+2];
        outputArray[grayOffset] = 0.21*r + 0.71*g + 0.07*b;
    }
}


void BGR2GRAY_wrapper(float *hostOutputArray, float *hostInputArray, int width, int height){
    float *deviceInputArray;
    float *deviceOutputArray;
    // Allocating memory for device
    cudaMalloc((void **)&deviceInputArray, width * height * CHANNELS * sizeof(float));
    cudaMalloc((void **)&deviceOutputArray, width * height * sizeof(float));
    // Copying host to device
    cudaMemcpy(deviceInputArray, hostInputArray, width * height * CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    //Calling the kernel
    dim3 threadBlocks(16,16);
    dim3 gridDimensions((width-1)/16 + 1, (height-1)/16 + 1);
    BGR2GRAY<<<gridDimensions, threadBlocks>>>(deviceOutputArray, deviceInputArray, width, height);
    // Copying device to host
    cudaMemcpy(hostOutputArray, deviceOutputArray, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deviceInputArray);
    cudaFree(deviceOutputArray);
    return;
}
