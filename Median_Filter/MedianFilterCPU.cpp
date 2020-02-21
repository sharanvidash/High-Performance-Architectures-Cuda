#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cstdlib>
#include <stdio.h>
#include "Bitmap.h"
#include<iostream>
#include <ctime>
#include "MedianFilter.h"
using namespace std;
void MedianFilterCPU(Bitmap* image, Bitmap* outputImage)
{
	
	for (int j = 0; j < image->Width(); j++)//column
	{
		for (int i = 0; i < image->Height(); i++)//row
		{
			//initialize arrays and count value
			int count = 0;
			unsigned char borderarray[9];
			int flag = 0;
		
			
			if (i == 0 || j == image->Width()-1 || j == 0 || i == image->Height()-1)
			{
				//Set the ouput pixel as 0 if it is a border element
				outputImage->SetPixel(j, i, 0);
			}
			else
			{
				//Setting the border array or the filter around the pixel
				borderarray[0] = image->GetPixel(j, i);
				borderarray[1] = image->GetPixel(j + 1, i);
				borderarray[2] = image->GetPixel(j - 1, i);
				borderarray[3] = image->GetPixel(j + 1, i + 1);
				borderarray[4] = image->GetPixel(j - 1, i + 1);
				borderarray[5] = image->GetPixel(j + 1, i - 1);
				borderarray[6] = image->GetPixel(j - 1, i - 1);
				borderarray[7] = image->GetPixel(j, i + 1);
				borderarray[8] = image->GetPixel(j, i - 1);

				//Bubble sorting of the border elements
				for (int o = 0; o < 9 - 1; o++)
				{// Last i elements are already in place  
					for (int p = 0; p < 9 - o - 1; p++)
					{
						if (borderarray[p] > borderarray[p + 1])
							swap(borderarray[p], borderarray[p + 1]);
					}
				}
				//Seting the median to be the pixel
				outputImage->SetPixel(j, i, borderarray[4]);
			}

		}
	}
}
