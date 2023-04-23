
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

