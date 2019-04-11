#include <stdio.h>


#define CSC(call) do {      \
    cudaError_t e = call;   \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(0);            \
    }                       \
} while(0)


__global__ void subKernel(double* a, double* b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Индекс нити
    int offset = gridDim.x * blockDim.x;              // кол-во блоков * размер блока
    while(idx < n) {
        b[idx] = a[idx] * a[idx];
        idx += offset;
    }
}


void sub(double* a, double* b, int n, int numBytes) {

    double* aDev = NULL;
    double* bDev = NULL;

    // Выделяем память на GPU
    CSC(cudaMalloc ( (void**)&aDev, numBytes ));
    CSC(cudaMalloc ( (void**)&bDev, numBytes ));

    // Задаем конфигурацию запуска нитей
    dim3 threads = 128;
    dim3 blocks = 128;

    CSC(cudaMemcpy ( aDev, a, numBytes, cudaMemcpyHostToDevice ));

    subKernel<<<blocks, threads>>> (aDev, bDev, n);

    // Копируем результат в память CPU
    CSC(cudaMemcpy ( b, bDev, numBytes, cudaMemcpyDeviceToHost ));

    // Освобождаем выделенную память
    CSC(cudaFree ( aDev ));
    CSC(cudaFree ( bDev ));
}


int main() {
    int n;
    scanf("%d", &n);

    int numBytes = n * sizeof(double);

    double* a = (double*) malloc(numBytes);
    double* b = (double*) malloc(numBytes);

    for (int i = 0; i < n; ++i)
        scanf("%lf", a + i);

    sub(a, b, n, numBytes);

    for (int i = 0; i < n; ++i)
        printf("%.10e ", b[i]);
    printf("\n");

    free(a);
    free(b);

    return 0;
}
