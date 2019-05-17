#include <stdio.h>
#include <stdlib.h>
typedef unsigned char uchar;
__global__ void calcGis(uchar* data, int n, int* height) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int offsetx = gridDim.x * blockDim.x;
        __shared__ int tmp[256];
        for(int i = threadIdx.x; i<256; i+=blockDim.x){
            tmp[i]=0;
        }
        __syncthreads();
        for(int i = idx; i < n; i += offsetx){
            atomicAdd(tmp+(int)data[i], 1);
        }
        __syncthreads();
        for(int i = threadIdx.x; i<256; i+=blockDim.x){
            atomicAdd(height + i, tmp[i]);
        }
}
__global__ void scan(int* height){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int forSwap,k;
        __shared__ int data[264];
        if(idx<256){
            data[idx+(idx>>5)]=height[idx];
            __syncthreads();
            for(k=1;k<256;k*=2){
                int j=idx*k*2+k-1;
                if(j+k<256){
                    data[((j+k)>>5)+(j+k)]+=data[j+(j>>5)];
                }
                __syncthreads();
            }
            data[((255)>>5)+(255)]=0;
            __syncthreads();
            for( k=256;k>1;k/=2){
                int j=k*(idx+1)-1;
                if(j<256){
                    forSwap=data[((j-k/2)>>5)+(j-k/2)];
                    data[((j-k/2)>>5)+(j-k/2)]=data[(j>>5)+j];
                    data[(j>>5)+j]=forSwap+data[(j>>5)+j];
                }
                __syncthreads();
            }
            __syncthreads();
            height[idx]+=data[idx+(idx>>5)];

        }
}
__global__ void outGis(uchar* data, int n, int* height) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int offsetx = gridDim.x * blockDim.x;
        uchar j = 0;
        for(int i = idx; i < n; i += offsetx){
            while(height[j] <= i){
                j++;
            }
            data[i] = j;
        }
}
int main() {
        int size = 256;
        int n;
        fread(&n, sizeof(int), 1, stdin);
        uchar* data = (uchar*) malloc(sizeof(uchar) * n);
        fread(data, sizeof(uchar), n, stdin);
        int *height1;
        uchar *data1;

        cudaMalloc(&data1, n * sizeof(uchar));
        cudaMemcpy(data1, data, n*sizeof(uchar), cudaMemcpyHostToDevice);
        cudaMalloc(&height1, size * sizeof(int));
        cudaMemset(height1, 0, size * sizeof(int));
        dim3 threads = 256;
        dim3 blocks = 256;

        calcGis<<<blocks, threads>>>(data1, n, height1);
        scan<<<blocks, threads>>>(height1);
        outGis<<<blocks, threads>>>(data1, n, height1);

        cudaMemcpy(data, data1, n * sizeof(uchar), cudaMemcpyDeviceToHost);

        cudaFree(height1);
        cudaFree(data1);
        fwrite(data, sizeof(uchar), n, stdout);
        free(data);
        return 0;
}
