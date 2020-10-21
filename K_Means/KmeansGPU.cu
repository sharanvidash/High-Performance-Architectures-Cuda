#include <cmath> // sqrt()
#include <ctime> // time(), clock()
#include <iostream> // cout, stream
#include <fstream>
#include "KMeans.h"
const int TILE_SIZE = 1024;
//constant memory is declared in global scope and is updated from the host and read by the device
__constant__ Vector2 Bd[3];
//Kernel to assign the clusters for each datapoint
__global__ void KmeansKernel(Datapoint* Ad, long n,int k)
{
	// Retrieve our global thread Id
	int idx = blockIdx.x *blockDim.x + threadIdx.x;
	// Assignment  
	if (idx < n)
	{
		int current_cluster_id = 0;
		float distance = Bd[0].distSq(Ad[idx].p);
		for (int dk = 0; dk < k; dk++)
		{
			if (Bd[dk].distSq(Ad[idx].p) < distance)
			{
				current_cluster_id = dk;
				distance = Bd[dk].distSq(Ad[idx].p);
			}
			
		}
		//Checking if the current cluster id is same as the previous to check for alterations
		if (current_cluster_id != Ad[idx].cluster)
		{
			Ad[idx].cluster = current_cluster_id;
			Ad[idx].altered = true;
		}
		else
			Ad[idx].altered = false;
	}
}
bool KMeansGPU(Datapoint* data, long n, Vector2* clusters, int k)
{
	// Error return value
	cudaError_t status;
	// Number of bytes in the vector
	int bytes = n * sizeof(Datapoint);
	// Pointers to the device arrays
	Datapoint *Ad, *Cd;
	
	// Allocate memory on the device to store each vector
	cudaMalloc((void**)&Ad, bytes);
	cudaMalloc((void**)&Cd, bytes);
	// Copy the host input data to the device
	cudaMemcpy(Ad, data, bytes, cudaMemcpyHostToDevice);
	
	// Specify the size of the grid and the size of the block
	dim3 dimBlock(TILE_SIZE); // Matrix is contained in a block
	dim3 dimGrid((int)ceil((float)n / (float)TILE_SIZE));
	int flag = false;
	//int current_cluster_id;
	while (flag==false)
	{
		flag = true;
		cudaMemcpyToSymbol(Bd, clusters, 3 * sizeof(Vector2));
		// Launch the kernel on a size-by-size block of threads
		KmeansKernel << <dimGrid, dimBlock >> >(Ad, n, k);
		// Wait for completion
		cudaThreadSynchronize();
		// Check for errors
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Kernel failed: " << cudaGetErrorString(status) <<
				std::endl;
			cudaFree(Ad);
			return false;
		}
		// Retrieve the result matrix
		cudaMemcpy(data, Ad, bytes, cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(clusters, Bd, 3 * sizeof(Vector2));


		// Updating the means in each iteration based on the data points in the cluster 

		for (int dk = 0; dk < k; dk++)
		{
			int num_points_cluster = 0;
			for (int dI = 0; dI < n; dI++)
			{
				if (data[dI].cluster == dk)
				{
					if (data[dI].altered == true)
						flag = false;
					num_points_cluster++;
					clusters[dk].x += data[dI].p.x;
					clusters[dk].y += data[dI].p.y;
				}

			}
			clusters[dk].x /= num_points_cluster;
			clusters[dk].y /= num_points_cluster;
		}
	}

	// Free device memory
	cudaFree(Ad);
	// Success
	return true;
}