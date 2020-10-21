**Median Filtering** 

**Abstract:** 

The program aims to achieve filtering of images using median filtering. Median filtering is used to filter images with black or white spots or salt and pepper noise. The algorithm consists of loading a skirt around each pixel of 3x3 and finding the median by using sorting. Since there are many calls to global memory, we use shared memory to increase speedup because it has low latency. There are 2 implementations made on the GPU, one using shared memory and the other without it. Both the implementations were checked for speedup and error in comparison with the CPU implementation. 

**Design Methodology:** 

Median Filtering is used to filter images with salt and pepper noise. The salt and pepper noise are pixels in the image which have either a high value or a very low value. This makes it easier to filter as they are not median values when surrounding a particular pixel. Median filtering uses this property to take a particular skirt around the pixel and sort the pixel values to choose the median value. The pixel’s value is replaced with the median value so there are no extreme values in the image. 

The CPU implementation was fairly simple as there was Bitmap.h to support writing and loading images. There were functions to support getting each pixel and writing each pixel. The elements around each pixel were stored in a separate array and was sorted using bubble sort. Bubble sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order. The pass through the list is repeated until the list is sorted. The median for the particular pixel filter is the middle element of the 9 elements in the borderarray[]. Borderarray[] is the filter around each pixel with 9 elements since it’s a 3x3 filter. Each pixel was set using SetPixel() to the median of its filter array.  

The GPU implementation varied in 2 types, one not using shared memory and the other using shared memory. Without shared memory, although parallel, the speedup is affected because of the latency caused by the many memory accesses to global memory. There a minimum of 9 global memory accesses not including the write to the output image. The bitmap image returns a character array when using InputImage.image, which was used to pass onto the GPU kernel as there are no support for bitmap within cuda and Bitmap.h has only host functions. The GPU launches a kernel with 2D blocks and a 2D grid. The TILE\_WIDTH was set to 32 for optimum usage of the SM resources when not using shared memory. Each thread calculates the median for each pixel. The borderarray[] was set without for loops and set directly. Bubble sort was used to sort the borderarray[] and borderarray[4] was chosen as the median for each pixel. The boundary conditions were checked by checking the global threadID in the x axis and y axis. If the pixel was a boundary pixel, the value was set to 0. Unsigned char was used to accommodate the pixel values. 

When shared memory was implemented, each block was copied into the shared memory for faster access. The issue with shared memory was that boundary elements in each block would not have elements to access. This problem was solved by using a padding of zeros around the block. The edges of the block had 0 to support accessing the filter elements. 

![](Lab05\_Report\_pdf(1).001.png)

*Figure 1: Example of Zero padding around the block* 

When using shared memory, we also have to consider banking problems. We overcame different threads accessing the same bank by setting a bigger shared memory block than the tile size. We used (TILE\_WIDTH +2) x (TILE\_WIDTH +2) dimension. The other boundary condition checks and sorting algorithm is same as the one used for GPU implementation without shared memory. 

**Results:** 

Lenna.bmp: 

![](Lab05\_Report\_pdf(1).002.png)

*Figure 2 Output for Lenna.bmp* 

![](Lab05\_Report\_pdf(1).003.png)

*Figure 3 Input Image for Lenna.bmp* 

![](Lab05\_Report\_pdf(1).004.png)

*Figure 4 Output Image for Lenna.bmp* 

Analysis: 

The output shows that the filtering is same for all three implementations with a marginal error on shared memory. The speedup is significantly more on the GPU implementation due to the problem being very parallel. 

Milkyway.bmp: 

![](Lab05\_Report\_pdf(1).005.png)

*Figure 5 Output for milkyway.bmp* 

![](Lab05\_Report\_pdf(1).006.png)

*Figure 6 Input Image for Milkyway.bmp* 

![](Lab05\_Report\_pdf(1).007.png)

*Figure 7 Output Image for Milkyway.bmp* 

Analysis: 

The output shows that the filtering is same for all three implementations with a marginal error on shared memory. The speedup is significantly more on the GPU implementation due to the problem being very parallel. 



|**Image** |**Widthx Height** |**CPU Time(ms)** |**GPU Global Time (ms)** |**GPU Global Speedup** |**GPU Shared Time (ms)** |**GPU Shared Speedup** |
| - | - | :- | :- | - | :- | :- |
|Lenna.bmp |52480 |67.5 |1.4 |48.214 |1.4 |48.214 |
|Milkyway.bmp |8000000 |9700 |69 |140.579 |70 |138.57 |


![](Lab05\_Report\_pdf(1).008.png)

GPU Global Memory Speedup

![](Lab05\_Report\_pdf(1).009.png)

GPU Shared Memory Speedup

Analysis: 

The speedup is very similar on both the shared and global memory implementations of the GPU. This could be attributed to a less efficient implementation of the shared memory and also declaring unused threads to prevent illegal memory accesses. 

**Conclusion:** 

Median Filtering was established on both the CPU and the GPU with 2 variations on the GPU. Speedup was observed on both the shared memory implementation and global memory implementation due to lower latency access and parallel nature of the problem. The speedup is not as expected because of an improper shared memory implementation. More speedup could be achieved ifvalues copied to the shared memory is done in an efficient way.  

