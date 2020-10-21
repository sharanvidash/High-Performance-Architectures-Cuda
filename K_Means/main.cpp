/************************************************************************/
// The purpose of this program is to perform K-MeansClustering.
//
// Author: <Your Name>
// Date: April 26, 2012
// Course: 0306-724 - High Performance Architectures
//
// File: main.cpp
/************************************************************************/

#include <cmath> // sqrt()
#include <ctime> // time(), clock()
#include <iostream> // cout, stream
#include <fstream>
#include "Kmeans.h"

#define ITERS 29
#define DATA_SIZE (1<<25)

// To reset the cluster data between runs.
void initializeClusters(Vector2* clusters)
{
	clusters[0].x = 0;
	clusters[0].y = 0;

	clusters[1].x = 1;
	clusters[1].y = 0;

	clusters[2].x = -1;
	clusters[2].y = 0;
}

/* Entry point for the program. 
   Performs k-means clustering on some sample data. */
int main() 
{
	// The data we want to operate on
	Datapoint* data = new Datapoint[DATA_SIZE];
	Datapoint* dataCPU = new Datapoint[DATA_SIZE];
    Datapoint* dataGPU = new Datapoint[ DATA_SIZE ];
	Vector2 clusters[3];

	std::cout << "Performing k-means clustering on " << DATA_SIZE << " values." << std::endl;
	
	// Fill up the example data using three gaussian distributed clusters.
	for (long i = 0; i < DATA_SIZE; i++) {
		int cluster = rand()%3;
		float u1 = (float)(rand()+1)/(float)RAND_MAX;
		float u2 = (float)(rand()+1)/(float)RAND_MAX;
		float z1 = sqrt(abs(-2 * log(u1))) * sin(6.283f*u2);
		float z2 = sqrt(abs(-2 * log(u1))) * cos(6.283f*u2);
		data[i].cluster = cluster; // ground truth
		switch (cluster) {
			case 0:
				data[i].p.x = z1;
				data[i].p.y = z2;
				break;
			case 1:
				data[i].p.x = 2 + z1 * 0.5f;
				data[i].p.y = 1 + z2 * 0.5f;
				break;
			case 2:
				data[i].p.x = -2 + z1 * 0.5f;
				data[i].p.y = 1 + z2 * 0.5f;
				break;
		}
	}


	float tcpu, tgpu;
	clock_t start, end;

	// Perform the host computations
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		memcpy(dataCPU, data, sizeof(Datapoint) * DATA_SIZE);
		initializeClusters(clusters);
		KMeansCPU(dataCPU, DATA_SIZE, clusters, 3);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

    long incorrect = 0;
	for (long i = 0; i < DATA_SIZE; i++)
		if (data[i].cluster != dataCPU[i].cluster) incorrect++;

	// Display the results
	std::cout << "Host Result took " << tcpu << " ms (" << (float)incorrect / (float)DATA_SIZE * 100 << "% misclassified)" << std::endl;
	for (int j = 0; j < 3; j++)
		std::cout << "Cluster " << j << ": " << clusters[j].x << ", " << clusters[j].y << std::endl;
    std::cout << std::endl;

	//=================================================================================================
	//Insert your code for the GPU computation here. You should follow the same 
	//format as the code provided for the CPU.
	//=================================================================================================
	// Perform Device Computations
	bool success = KMeansGPU(dataGPU, DATA_SIZE, clusters, 3);
	if (!success)
	{
		std::cout << "\n * device error! * \n" << std::endl;
		while (true);
		return -1;
	}

	start = clock();
	for (int i = 0; i < ITERS; i++) {
		memcpy(dataGPU, data, sizeof(Datapoint) * DATA_SIZE);
		initializeClusters(clusters);
		KMeansGPU(dataGPU, DATA_SIZE, clusters, 3);
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

	 incorrect = 0;
	for (long i = 0; i < DATA_SIZE; i++)
		if (data[i].cluster != dataGPU[i].cluster) incorrect++;

	// display the results
	std::cout << "device result took " << tgpu << " ms (" << (float)incorrect / (float)DATA_SIZE * 100 << "% misclassified)" << std::endl;
	for (int j = 0; j < 3; j++)
		std::cout << "cluster " << j << ": " << clusters[j].x << ", " << clusters[j].y << std::endl;
	std::cout << std::endl;
	
 //  
	//Write the results to a file.
    std::ofstream outfile("results.csv");
	outfile << "x,y,Truth,CPU,GPU" << std::endl;
	for (long i = 0; i < DATA_SIZE; i++) {
		outfile << data[i].p.x << "," << data[i].p.y << "," << data[i].cluster << "," << dataCPU[i].cluster << "," << dataGPU[i].cluster << "\n";
	}
	outfile.close();

	delete[] data;
	delete[] dataCPU;
    delete[] dataGPU;

	// Success
	return 0;
}
