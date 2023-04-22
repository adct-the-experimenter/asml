
#include "kernel.cuh"


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
