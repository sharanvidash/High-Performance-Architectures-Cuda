
#include <cmath> // sqrt()
#include <ctime> // time(), clock()
#include <iostream> // cout, stream
#include <fstream>
#include "KMeans.h"


void KMeansCPU(Datapoint* data, long n, Vector2* clusters, int k)
{
	//Flag is used to determine the point to stop the iterations
	bool flag=false;
	while (flag==false)
	{
		flag = true;
		// Assignment  
		for (int dI = 0; dI < n; dI++)
		{
			int current_cluster_id = 0;
			float distance = clusters[0].distSq(data[dI].p);
			for (int dk = 0; dk < k; dk++)
			{
				if (clusters[dk].distSq(data[dI].p) < distance)
				{
					current_cluster_id = dk;
					distance = clusters[dk].distSq(data[dI].p);
				}
				
			}
			//Checking if the current cluster id is same as the previous to check for alterations
			if (current_cluster_id != data[dI].cluster)
			{
				data[dI].cluster = current_cluster_id;
				flag = false;
				data[dI].altered = true;
			}
		}

		// Updating the means in each iteration based on the data points in the cluster
		
		for (int dk = 0; dk < k; dk++)
		{
			int num_points_cluster = 0;
			for (int dI = 0; dI < n; dI++)
			{
				if (data[dI].cluster == dk)
				{
					num_points_cluster++;
					clusters[dk].x += data[dI].p.x;
					clusters[dk].y += data[dI].p.y;
				}

			}
			clusters[dk].x /= num_points_cluster;
			clusters[dk].y /= num_points_cluster;
		}
	} 
	
}