#include <cstdio>
#include <omp.h>

int main()
{
	int n = 50000039;
	double dx = 1./n;
	long double pi = 0;
	int cpu_threads = omp_get_max_threads();
	int num_threads = cpu_threads;
	omp_set_num_threads(num_threads);
	printf("CPU threads: %d\n", cpu_threads);
	printf("Using %d threads\n", num_threads);
	double start = omp_get_wtime();
	#pragma omp parallel for reduction(+:pi)
	for (int i=0; i<n; i++)
	{
		double x = (i+0.5) *dx;
		pi += 4.0/(1.0+x*x) *dx;
	}
	double end = omp_get_wtime();
	printf("Time: %.4f seconds\n", end-start);
	printf("%.15Lf\n",pi);
}
