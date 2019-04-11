#include <stdio.h>
#include <assert.h>


#define CSC(call) do {				\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "CUDA error %s:%d message: %s\n", __FILE__, __LINE__,	\
				cudaGetErrorString(res));	\
		exit(0);							\
	}										\
} while(0)


//__host__ __device__ int add(int a, int b){
//	return a + b;
//}

__global__ void kernel(int *arr, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;
	while(idx < n) {
//		assert(idx < 100);
		arr[idx] *= 2;
		idx += offset;
	}
}

int main() {
	int i, n = 100000000;
	int *arr = (int *)malloc(sizeof(int) * n);
	for(i = 0; i < n; i++)
		arr[i] = i;

	int *dev_arr;
	CSC(cudaMalloc(&dev_arr, sizeof(int) * n));
	CSC(cudaMemcpy(dev_arr, arr, sizeof(int) * n, cudaMemcpyHostToDevice));

	float time;
	cudaEvent_t start, stop;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	CSC(cudaEventRecord(start, 0));

	kernel<<<256, 256>>>(dev_arr, n);
	CSC(cudaGetLastError());

	CSC(cudaEventRecord(stop, 0));
	CSC(cudaEventSynchronize(stop));
	CSC(cudaEventElapsedTime(&time, start, stop));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));
	printf("time = %f\n", time);

	CSC(cudaMemcpy(arr, dev_arr, sizeof(int) * n, cudaMemcpyDeviceToHost));

	for(i = n - 100; i < n; i++)
		printf("%d ", arr[i]);
	printf("\n");

	CSC(cudaFree(dev_arr));
	free(arr);
	return 0;
}
