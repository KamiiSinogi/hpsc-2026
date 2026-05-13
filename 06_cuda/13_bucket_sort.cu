#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_init(int* bucket, int range)
{
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<range) bucket[idx] = 0;
}

__global__ void count(int* key, int* bucket, int n, int range)
{
	extern __shared__ int block_bucket[];
	for(int i=threadIdx.x; i<range; i+=blockDim.x)
	{
		block_bucket[i] = 0;
	}
	__syncthreads();
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<n) atomicAdd(&block_bucket[key[idx]], 1);
	__syncthreads();
	for(int i=threadIdx.x; i<range; i+=blockDim.x)
	{
		atomicAdd(&bucket[i], block_bucket[i]);
	}
}

__global__ void sum(int* bucket, int* offset, int range)
{
	offset[0]=0;
	for(int i=1; i<range; i++)
	{
		offset[i]=offset[i-1]+bucket[i-1];
	}
}

__global__ void sort(int* key, int* bucket, int* offset, int n, int range)
{
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<range)
	{
		int start=offset[idx], end=offset[idx]+bucket[idx];
		for(int i=start; i<end; i++) key[i]=idx;
	}
}

bool check_sorted(std::vector<int>& key, std::vector<int>& cnt)
{
	for (int i=0; i<key.size(); i++)
	{
		if (cnt[key[i]] == 0 || (i < key.size()-1 && key[i] > key[i+1])) return 0;
		cnt[key[i]]--;
	}
	return 1;
}

int main()
{
	int n = 500039;
	int range = 39;
	std::vector<int> key(n), cnt(range, 0);
	for (int i=0; i<n; i++)
	{
		key[i]=rand()%range;
//		printf("%d ",key[i]);
		cnt[key[i]]++;
	}
//	printf("\n");

	int threads=256;
	int *cuda_key, *cuda_bucket, *cuda_offset;
	cudaMalloc(&cuda_key, n*sizeof(int));
	cudaMalloc(&cuda_bucket, range*sizeof(int));
	cudaMalloc(&cuda_offset, range*sizeof(int));
	cudaMemcpy(cuda_key, key.data(), n*sizeof(int), cudaMemcpyHostToDevice);
	bucket_init<<<(range+threads-1)/threads,threads>>>(cuda_bucket, range);
	count<<<(n+threads-1)/threads,threads,range*sizeof(int)>>>(cuda_key, cuda_bucket, n, range);
	sum<<<1,1>>>(cuda_bucket, cuda_offset, range);
	sort<<<(range+threads-1)/threads,threads>>>(cuda_key, cuda_bucket, cuda_offset, n, range);
	cudaMemcpy(key.data(), cuda_key, n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// for (int i=0; i<n; i++)
	// {
	// 	printf("%d ",key[i]);
	// }
	// printf("\n");

	if(check_sorted(key, cnt)) printf("Sorted\n");
	else printf("Not sorted\n");
}
