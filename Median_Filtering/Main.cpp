#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cstdlib>
#include <stdio.h>
#include "Bitmap.h"
#include<iostream>
#include <ctime>
#include "MedianFilter.h"
//Function to compare bitmaps
int CompareBitmaps(Bitmap* inputA, Bitmap* inputB)
{
	int numpixels = 0, Width, Height;


	if (inputA->Width() == inputB->Width() && inputA->Height() > inputB->Height())
	{

		for (int i = 0; i < Width; i++)
		{
			for (int j = 0; j < Height; j++)
			{
				if (inputA->GetPixel(i, j) != inputB->GetPixel(i, j))
					numpixels++;
			}
		}

		return numpixels;
	}
	else
	{
		printf("Unequal images");
		return 0;
	}
}
void main()
{
	const int ITERS = 1;
	// Timing data
	float tcpu, tgpu;
	clock_t start, end;
	// Allocate the four bitmaps and load the images
	Bitmap A, B,Ccpu,Cgpu,Cgpushared;
	A.Load("milkyway.bmp");
	B.Load("Lenna.bmp");
	Ccpu.Load("Lenna.bmp");
	Cgpu.Load("Lenna.bmp");
	Cgpushared.Load("Lenna.bmp");
	Ccpu.Save("Output.bmp");
	int size = B.Width() * B.Height();
	


	std::cout << "Operating on an image of size " << B.Width()<<" X "<< B.Height() << std::endl;
//Host Median Filtering
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		MedianFilterCPU(&B, &Ccpu);
		
	}
	end = clock();
	Ccpu.Save("Output2.bmp");
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout << "Average Time Per CPU Iteration is: " << tcpu << " ms:" << std::endl;

	//Device MEdian Filtering Using shared memory
	// Perform one warm-up pass and validate
	bool success = MedianFilterGPU(&B, &Cgpushared, 1);
	if (!success) {
		std::cout << "\n * Device error! * \n" << std::endl;
	}
	// And now time it
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		MedianFilterGPU(&B, &Cgpushared, 1);
	}
	Cgpushared.Save("Output3.bmp");

	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout << "Average Time per GPU Iteration with shared memory is : " << tgpu << " ms:" << std::endl;
	printf("%d differing pixels compared to CPU algorithm \n", CompareBitmaps(&Ccpu, &Cgpushared));
	//Device MEdian Filtering Using shared memory
	// Perform one warm-up pass and validate
	 success = MedianFilterGPU(&B, &Cgpu, 0);
	if (!success) {
		std::cout << "\n * Device error! * \n" << std::endl;
	}
	// And now time it
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		MedianFilterGPU(&B, &Cgpu, 0);
	}
	Cgpushared.Save("Output4.bmp");

	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout << "Average Time per GPU Iteration with global memory is : " << tgpu << " ms:" << std::endl;
	printf("%d differing pixels compared to CPU algorithm \n", CompareBitmaps(&Ccpu, &Cgpu));

	// Compare the results for correctness
	float sum = 0, delta = 0;
	char* Ccpucheck = Ccpu.image;
	char* Cgpucheck = Cgpu.image;
	for (int i = 0; i < size; i++) {
		delta += (Ccpucheck[i] - Cgpucheck[i]) * (Ccpucheck[i] - Cgpucheck[i]);
		sum += (Ccpucheck[i] * Cgpucheck[i]);
	}
	//std::cout << "Median Filtering Speedup " << (tcpu / tgpu) << std::endl;
	float L2norm = sqrt(delta / sum);
	std::cout << "Used " << ITERS<<" iterations" << "\n" << std::endl;
	std::cout << "Median Filtering error: " << L2norm << "\n" << std::endl;
	
}
