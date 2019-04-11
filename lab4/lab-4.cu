#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

using namespace std;


void zzz2(int c){
    if(c<10){
        char a=c+'0';
        printf("%c",a);
    }else{
        char a=c-10+'a';
        printf("%c",a);
    }
}
void zzz1(uchar3 a){
    int x=a.x;
    int y=a.y;
    int z=a.z;
    zzz2(x/16);
    zzz2(x%16);
    zzz2(y/16);
    zzz2(y%16);
    zzz2(z/16);
    zzz2(z%16);
}
void zzz3(uchar4 a){
    int x=a.x;
    int y=a.y;
    int z=a.z;
    zzz2(x/16);
    zzz2(x%16);
    zzz2(y/16);
    zzz2(y%16);
    zzz2(z/16);
    zzz2(z%16);
}

__device__ double SQRT(double A){
        return sqrt(A);
}
__device__ double POW(double a){
    return  a*a;
}
#define CSC(call) do {      \
    cudaError_t e = call;   \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n"\
        , __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(0);            \
    }                       \
} while(0)

#define RGBToWB(r, g, b)  0.299*r + 0.587*g + 0.114*b
#define chek(a) a>255.1?255.1:a


struct abs_max {
    __host__ __device__ bool operator()(double a, double b) {
        return abs(a) < abs(b);
    }
};
__global__ void my_swap(double* data,double* E, int n,int x,int y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Индекс нити
    int offset = gridDim.x * blockDim.x;              // кол-во блоков * размер блока
    int i;
    double tmp;
    for(i=idx;i<n;i+=offset){
        tmp=data[i*n+x];
        data[i*n+x]=data[i*n+y];
        data[i*n+y]=tmp;
        tmp=E[i*n+x];
        E[i*n+x]=E[i*n+y];
        E[i*n+y]=tmp;
    }
}
__global__ void normalization(double* data,double* E, int n,int i){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Индекс нити
    int offset = gridDim.x * blockDim.x;              // кол-во блоков * размер блока
    int j;
    double tmp=data[i*n+i];
    for(j=idx;j<n;j+=offset){
        if(j!=i)
        data[j*n+i]/=tmp;
        E[j*n+i]/=tmp;
    }
}
__global__ void kernel(double* data,double* E, int n,int x) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int idy = threadIdx.y + blockDim.y * blockIdx.y;
        int offsetx = blockDim.x * gridDim.x;
        int offsety = blockDim.y * gridDim.y;
        int i, j;
        for(i = idx; i < n; i += offsetx){
                for(j = idy; j < n; j += offsety){
                    if(i!=x){
                        //a*n b
                        E[j*n+i]-=data[x*n+i]*E[j*n+x];
                        if(j!=x)
                        data[j*n+i]-=data[x*n+i]*data[j*n+x];
                    }
                }

        }
}

int main() {
        abs_max zzz;
        int n;
        scanf("%d", &n);
        //fprintf(stderr,"%d\n",n);
        double* data = (double*) malloc( n*n*sizeof(double) );
        double* E = (double*) malloc( n*n*sizeof(double) );
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i){
                scanf("%lf", &data[i * n + j]);
                //fprintf(stderr,"%.10e ",data[i * n + j]);
                if(i==j){
                    E[i * n + j]=1;
                }else{
                    E[i * n + j]=0;
                }
            }
            //fprintf(stderr,"\n");
        }
        double *dev_data,*dev_E;
        CSC( cudaMalloc( &dev_data, n*n*sizeof(double) ) );
        CSC( cudaMemcpy( dev_data, data, n*n*sizeof(double), cudaMemcpyHostToDevice ) );
        CSC( cudaMalloc( &dev_E, n*n*sizeof(double) ) );
        CSC( cudaMemcpy( dev_E, E, n*n*sizeof(double), cudaMemcpyHostToDevice ) );
        for(int i=0;i<n;i++){
            if(i!=n-1){
                thrust::device_ptr<double> ptr_data = thrust::device_pointer_cast(dev_data);
                thrust::device_ptr<double> max_elem_ref = thrust::max_element(
                ptr_data + i * n + i,
                ptr_data +i * n+ n,
                zzz );
                int max_pos = max_elem_ref - (ptr_data + i * n);
                //fprintf(stderr,"%d %d\n",max_pos,i);
                if(max_pos!=i){
                    my_swap <<<dim3(1024), dim3(1024)>>> (dev_data,dev_E, n, i, max_pos);
                }
            }
            normalization <<<dim3(1024), dim3(1024)>>>(dev_data,dev_E,n,i);
            kernel<<<dim3(32, 32), dim3(32, 32)>>>(dev_data,dev_E,n,i);
        }
        //normalization <<<dim3(1024), dim3(1024)>>>(dev_data,dev_E,n,n-1);
        CSC( cudaMemcpy( E, dev_E,  n*n*sizeof(double), cudaMemcpyDeviceToHost ) );

        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i){
                //fprintf(stderr,"%.10e ",E[i * n + j]);
                printf("%.10e ",E[i * n + j]);
            }
            //fprintf(stderr,"\n");
            printf("\n");
        }
        CSC(cudaFree ( dev_E ));
        CSC(cudaFree ( dev_data ));
         free(data);
         free(E);
        return 0;
}
