#include <stdio.h>
#include <stdlib.h>
using namespace std;


const int tread_size=1024;
#define CSC(call) do {      \
    cudaError_t e = call;   \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n"\
        , __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(0);            \
    }                       \
} while(0)
#define f(n) (n)+((n)>>5)
#define f1(n,step) (((n)/step-1)*step+step-1)
__global__ void kernel1(int* height, long long sz,long long step) {
        long long idx = threadIdx.x + blockIdx.x * blockDim.x;
        long long offsetx = gridDim.x * blockDim.x;
        long long j,i,k;
        __shared__ int data[f(tread_size)];
        for(i=idx*step+step-1;i<sz;i+=offsetx*step){
            __syncthreads();
            data[f(threadIdx.x)]=height[i];
            for(k=1;k<tread_size;k*=2){
                __syncthreads();
                for( j=threadIdx.x*k*2+k-1;j+k<tread_size;j+=blockDim.x*k*2){
                    data[f(j+k)]+=data[f(j)];
                }
                __syncthreads();
            }
            data[f(tread_size-1)]=0;
            __syncthreads();
            for( k=tread_size;k>1;k/=2){
                __syncthreads();
                for(j=k-1+threadIdx.x*k;j<tread_size;j+=blockDim.x*k){
                	int tmp=data[f(j-k/2)];
                    data[f(j-k/2)]=data[f(j)];
                    data[f(j)]=tmp+data[f(j)];
                }
                __syncthreads();
            }
            __syncthreads();
            height[i]+=data[f(threadIdx.x)];
        }

        /*for (int j = i - 1; j + i < sz; j += i * 2) {
            height[j + i] += height[j];
            }*/8
        
}
__global__ void kernel2(int* height, long long sz,long long step) {
    long long idx = threadIdx.x + blockIdx.x * blockDim.x;
    long long offsetx = gridDim.x * blockDim.x;
    long long i;
    for(i=idx*step/tread_size+step+step/tread_size-1;i<sz;i+=offsetx*step/tread_size){
        if(i%step!=step-1){
            height[i]+=height[f1(i,step)];
        }
    }
}
__global__ void DTH(int* data, int n,int* height) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int offsetx = gridDim.x * blockDim.x;
        int i;
        for(i = idx; i < n; i += offsetx){
            //__threadfence_system();
            atomicAdd(height+data[i],1);
            //printf("%d %d\n",height[data[i]],data[i]);
        }
}
__global__ void HTD(int* data,int* in, int n,int* height) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int offsetx = gridDim.x * blockDim.x;
        int i;
        for(i = idx; i < n; i += offsetx){
            //__threadfence_system();

            //__threadfence();

            //printf("%d %d\n",height[in[i]],in[i]);
            data[atomicSub(height + in[i],1)-1]=in[i];
            //__threadfence();
        }
}
int main() {
        long long sz=1<<24 ;
        int n;
        //cin >> n;
        fread(&n, sizeof(int), 1, stdin);
        //int* in=(int*) malloc( sizeof(int) * n);
        int* data=(int*) malloc( sizeof(int) * n);
        //int* height=(int* ) malloc( sizeof(int) * n);
        fread(data, sizeof(int), n, stdin);
        /*for(int i=0;i<n;i++){
            cin >> data[i];
        }*/
        int *gpu_in,*gpu_data,*gpu_height;

        fprintf(stderr, "n=%d\n",n);
        for(int i=0;i<min(n,100);i++){
            fprintf(stderr, "%d ",data[i]);
        }
        fprintf(stderr, "\n");
        CSC( cudaMalloc( &gpu_in, n*sizeof(int) ) );
        CSC( cudaMalloc( &gpu_data, n*sizeof(int) ) );
        CSC( cudaMemcpy( gpu_in, data, n*sizeof(int), cudaMemcpyHostToDevice ) );
        CSC( cudaMalloc( &gpu_height, sz*sizeof(int)));
        CSC( cudaMemset(gpu_height,0,sz*sizeof(int)));
        dim3 threads = tread_size;
        dim3 blocks = tread_size;
        /*for(i = 0; i < n; i++){
                height[data[i]]++;
        }*/

        DTH<<<blocks,threads>>>(gpu_in,n,gpu_height);
        long long i=1;
        for(;i<sz;i*=tread_size)
            kernel1<<<blocks,threads>>>(gpu_height,sz,i);
        /*for (int j = i - 1; j + i < sz; j += i * 2) {
        height[j + i] += height[j];
        }*/
        //__threadfence_system();

        for(;i>1;i/=tread_size)
        kernel2<<<blocks,threads>>>(gpu_height,sz,i);
        /*for(j=i-1;j+i/2<sz-1;j+=i){
            height[j+i/2]+=height[j];
        }*/
        //__threadfence_system();
        /*for(i = idx; i < n; i += offsetx){
                in[--height[in[i]]]=data[i];
        }*/
        HTD<<<blocks,threads>>>(gpu_data,gpu_in,n,gpu_height);
        CSC( cudaMemcpy( data,gpu_data,  n*sizeof(int), cudaMemcpyDeviceToHost ) );
        /*for(int i=0;i<n;i++){
            cout << data[i]<<" ";
        }
        cout << endl;*/
        fwrite(data, sizeof(int), n, stdout);
        /*for(int i=0;i<n;i++){
            fprintf(stderr, "%d ",data[i]);
        }
        fprintf(stderr, "\n");*/
        CSC(cudaFree ( gpu_height ));
        CSC(cudaFree ( gpu_in ));
        CSC(cudaFree ( gpu_data ));
         free(data);
        return 0;
