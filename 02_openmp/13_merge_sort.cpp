#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <omp.h>


int para_limit = 1039;

void merge(std::vector<int>& vec, int begin, int mid, int end)
{
	std::vector<int> tmp(end-begin+1);
	int left = begin;
	int right = mid+1;
	for (int i=0; i<tmp.size(); i++)
	{ 
		if (left > mid)
			tmp[i] = vec[right++];
		else if (right > end)
			tmp[i] = vec[left++];
		else if (vec[left] <= vec[right])
			tmp[i] = vec[left++];
		else
			tmp[i] = vec[right++]; 
	}
	for (int i=0; i<tmp.size(); i++)
		vec[begin++] = tmp[i];
}

void merge_sort(std::vector<int>& vec, int begin, int end)
{
	if(begin < end) 
	{
		int mid = (begin+end)/2;
		if(end-begin < para_limit)
		{
			merge_sort(vec, begin, mid);
			merge_sort(vec, mid+1, end);
			merge(vec, begin, mid, end);
			return;
		}
		#pragma omp task shared(vec) firstprivate(begin, mid)
			merge_sort(vec, begin, mid);
		#pragma omp task shared(vec) firstprivate(mid, end)
			merge_sort(vec, mid+1, end);
		#pragma omp taskwait
			merge(vec, begin, mid, end);
	}
}

bool check_sorted(std::vector<int>& vec)
{
	for (int i=1; i<vec.size(); i++)
	{
		if (vec[i-1] > vec[i])
		{
			printf("Not sorted\n");
			return 0;
		}
	}
	return 1;
}

int main()
{
	int n = 100000039;
	std::vector<int> vec(n);

	srand(time(0));
	for (int i=0; i<n; i++)
	{
		vec[i] = rand() %(10*n);
		// printf("%d ",vec[i]);
	}
	// printf("\n");
	printf("Array length: %d\n",n);

	int cpu_threads = omp_get_max_threads();
	int num_threads = cpu_threads;
	omp_set_num_threads(num_threads);
	printf("CPU threads: %d\n", cpu_threads);
	printf("Using %d threads\n", num_threads);

	double start = omp_get_wtime();
	#pragma omp parallel
		#pragma omp single
			merge_sort(vec, 0, n-1);
	double end = omp_get_wtime();
	printf("Time: %.4f seconds\n", end-start);

	// for (int i=0; i<n; i++)
	// 	printf("%d ",vec[i]);
	// printf("\n");

	if (check_sorted(vec))
		printf("Sorted\n");
}