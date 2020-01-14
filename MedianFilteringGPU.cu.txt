#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "MedianFIlter.h"
#include "Bitmap.h"
using namespace std;
const int TILE_WIDTH = 31;
// GPU Kernel to perform Median Filtering using global memory
__global__ void MedianFilterKernel(unsigned char * Ad, unsigned char * Bd, int Width,int Height)
{
	// Retrieve our global thread Id
	int j = blockIdx.y * blockDim.x + threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.x;
	unsigned char borderarray[9];
	if (i < Width && j < Height)
	{
		if (i == Width - 1 || j == Height - 1 || j == 0 || i == 0)
		{
			//Set the ouput pixel as 0 if it is a border element
			Bd[j*Width + i] = 0;
		}
		else
		{
			//Setting the border array or the filter around the pixel
			borderarray[0] = Ad[j * Width + i];
			borderarray[1] = Ad[(j + 1) * Width + i];
			borderarray[2] = Ad[(j - 1) * Width + i];
			borderarray[3] = Ad[(j + 1) * Width + i + 1];
			borderarray[4] = Ad[(j - 1) * Width + i + 1];
			borderarray[5] = Ad[(j + 1) * Width + i - 1];
			borderarray[6] = Ad[(j - 1) * Width + i - 1];
			borderarray[7] = Ad[j * Width + i + 1];
			borderarray[8] = Ad[j * Width + i - 1];
			unsigned char temp;
			//Bubble sorting of the border elements
			for (int o = 0; o < 9 - 1; o++)
			{// Last i elements are already in place  
				for (int p = 0; p < 9 - o - 1; p++)
				{
					if (borderarray[p] > borderarray[p + 1])
					{
						temp = borderarray[p];
						borderarray[p] = borderarray[p + 1];
						borderarray[p + 1] = temp;

					}
				}

			}
			//Seting the median to be the pixel
			Bd[j*Width + i] = borderarray[4];
		}
	}

}
__global__ void MedianFilterKernelShared(unsigned char * Ad, unsigned char * Bd, int Width, int Height)
{
	// Retrieve our global thread Id
	int j = blockIdx.y * blockDim.x + threadIdx.y;
	int i = blockIdx.x * blockDim.y + threadIdx.x;
	__shared__ unsigned char shared[(TILE_WIDTH + 2)][(TILE_WIDTH + 2)];
	if (i < Width && j < Height)
	{
		if (i == Width-1 || j == Height-1 || j == 0 || i == 0)
		{
			//Set the ouput pixel as 0 if it is a border element
			Bd[j * Width + i] = 0;
		}
		else
		{
			//Initialize with zero around the block(zero padding) to avoid illegal memory access
			if (threadIdx.x == 0)
				shared[threadIdx.x][threadIdx.y + 1] = 0;
			else if (threadIdx.x == TILE_WIDTH-1)
				shared[threadIdx.x + 2][threadIdx.y + 1] = 0;
			if (threadIdx.y == 0) {
				shared[threadIdx.x + 1][threadIdx.y] = 0;
				if (threadIdx.x == 0)
					shared[threadIdx.x][threadIdx.y] = 0;
				else if (threadIdx.x == TILE_WIDTH-1)
					shared[threadIdx.x + 2][threadIdx.y] = 0;
			}
			else if (threadIdx.y == TILE_WIDTH-1) {
				shared[threadIdx.x + 1][threadIdx.y + 2] = 0;
				if (threadIdx.x == TILE_WIDTH-1)
					shared[threadIdx.x + 2][threadIdx.y + 2] = 0;
				else if (threadIdx.x == 0)
					shared[threadIdx.x][threadIdx.y + 2] = 0;
			}
			__syncthreads();
			//Setup pixel values
			shared[threadIdx.x + 1][threadIdx.y + 1] = Ad[j * Width + i];
			//Check for boundary conditions within block and set the pixel values
			if (threadIdx.x == 0 && (i > 0))
				shared[threadIdx.x][threadIdx.y + 1] = Ad[j * Width + (i - 1)];
			else if (threadIdx.x == TILE_WIDTH-1 && (i < Width - 1))
				shared[threadIdx.x + 2][threadIdx.y + 1] = Ad[j * Width + (i + 1)];
			if (threadIdx.y == 0 && (j > 0)) {
				shared[threadIdx.x + 1][threadIdx.y] = Ad[(j - 1) * Width + i];
				if (threadIdx.x == 0)
					shared[threadIdx.x][threadIdx.y] = Ad[(j - 1) * Width + (i - 1)];
				else if (threadIdx.x == TILE_WIDTH-1)
					shared[threadIdx.x + 2][threadIdx.y] = Ad[(j - 1) * Width + (i + 1)];
			}
			else if (threadIdx.y == TILE_WIDTH-1 && (j < Height - 1)) {
				shared[threadIdx.x + 1][threadIdx.y + 2] = Ad[(j + 1) * Width + i];
				if (threadIdx.x == TILE_WIDTH-1)
					shared[threadIdx.x + 2][threadIdx.y + 2] = Ad[(j + 1) * Width + (i + 1)];
				else if (threadIdx.x == 0)
					shared[threadIdx.x][threadIdx.y + 2] = Ad[(j + 1) * Width + (i - 1)];
			}
			
			
		  //Wait for all threads to be done.
			__syncthreads();
			//Setup the filter.
			unsigned char borderarray[9] = { shared[threadIdx.x][threadIdx.y], shared[threadIdx.x + 1][threadIdx.y], shared[threadIdx.x + 2][threadIdx.y],
						   shared[threadIdx.x][threadIdx.y + 1], shared[threadIdx.x + 1][threadIdx.y + 1], shared[threadIdx.x + 2][threadIdx.y + 1],
						   shared[threadIdx.x][threadIdx.y + 2], shared[threadIdx.x + 1][threadIdx.y + 2], shared[threadIdx.x + 2][threadIdx.y + 2] };

			unsigned char temp;
			//Bubble sorting of the border elements
			for (int o = 0; o < 9 - 1; o++)
			{// Last i elements are already in place  
				for (int p = 0; p < 9 - o - 1; p++)
				{
					if (borderarray[p] > borderarray[p + 1])
					{
						temp = borderarray[p];
						borderarray[p] = borderarray[p + 1];
						borderarray[p + 1] = temp;

					}
				}

			}
			//Seting the median to be the pixel
			Bd[(j)*Width + i] = borderarray[4];
		}
	}

}


// Helper Function to run The GPU implementations
bool MedianFilterGPU(Bitmap* image, Bitmap* outputImage, bool sharedoryUse)

{
	// Error return value
	cudaError_t status;
	// Number of bytes in the image
	int size = image->Height() * image->Width();
	int bytes = image->Height() * image->Width() * sizeof(char);
	// Pointers to the device arrays
	unsigned char* Ad, * Bd;
	// Allocate memory on the device to store each character matrix
	cudaMalloc((void**)& Ad, bytes);
	cudaMalloc((void**)& Bd, bytes);
	// Copy the host input data to the device
	cudaMemcpy(Ad, image->image, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, outputImage->image, bytes, cudaMemcpyHostToDevice);
	// Specify the size of the grid and the size of the block
	dim3 dimGrid((int)ceil((float)image->Width() / (float)TILE_WIDTH), (int)ceil((float)image->Height() / (float)TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	// Launch the kernel on a size-by-size block of threads
	int Width = image->Width(), Height = image->Height();
	if (sharedoryUse == 0)
	{
		//Launching the Global memory implementation kernel
		MedianFilterKernel << <dimGrid, dimBlock >> > (Ad, Bd, Width, Height);
	}
	else
	{
		//Launching the Shared memory implementation kernel
		MedianFilterKernelShared << <dimGrid, dimBlock >> > (Ad, Bd, Width, Height);
	}
	// Wait for completion
	cudaThreadSynchronize();
	// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed: " << cudaGetErrorString(status) <<
			std::endl;
		cudaFree(Ad);
		cudaFree(Bd);
		return false;
	}
	// Retrieve the result matrix
	cudaMemcpy(outputImage->image, Bd, bytes, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	// Success
	return true;
}