#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <omp.h>

int range_limit = 39;

bool check_sorted(std::vector<int>& key, std::vector<int>& count)
{
	for (int i=0; i<key.size(); i++)
	{
		if (count[key[i]] == 0 || (i < key.size()-1 && key[i] > key[i+1]))
		{
			printf("Not sorted\n");
			return 0;
		}
		count[key[i]]--;
	}
	return 1;
}

int main()
{
	int n = 1000000039;
	int range = 100039;
	std::vector<int> key(n);
	std::vector<int> count(range,0);

	srand(time(0));
	for (int i=0; i<n; i++)
	{
		key[i] = rand() % range;
		// printf("%d ",key[i]);
		count[key[i]]++;
	}
	// printf("\n");
	printf("Array length: %d\n",n);

	int cpu_threads = omp_get_max_threads();
	int num_threads = cpu_threads;
	omp_set_num_threads(num_threads);
	printf("CPU threads: %d\n", cpu_threads);
	printf("Using %d threads\n", num_threads);

	std::vector<int> bucket(range,0);
	std::vector<int> offset(range,0);
	int *bucket_data = bucket.data();

	double start = omp_get_wtime();
	#pragma omp parallel for reduction(+:bucket_data[:range])
		for (int i=0; i<n; i++)
			bucket_data[key[i]]++;
	for (int i=1; i<range; i++) 
		offset[i] = offset[i-1] + bucket[i-1];
	#pragma omp parallel for
		for (int i=0; i<range; i++)
		{
			int j = offset[i];
			for (int k=bucket[i]; k>0; k--)
				key[j++] = i;
		}
	double end = omp_get_wtime();
	printf("Time: %.4f seconds\n", end-start);

	// for (int i=0; i<n; i++)
	// 	printf("%d ",key[i]);
	// printf("\n");

	if (check_sorted(key,count))
		printf("Sorted\n");
}
