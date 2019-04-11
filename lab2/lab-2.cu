#include <stdio.h>
#include <stdlib.h>
#include <cmath>

using namespace std;

__device__ double f(double A){
        return sqrt(A);
}
#define CSC(call) do {      \
    cudaError_t e = call;   \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n"\
        , __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(0);            \
    }                       \
} while(0)

#define RGBToWB(r, g, b)  0.299*r + 0.587*g + 0.114*b;
#define chek(a) a>255.1?255.1:a;

texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4 *dst, int w, int h) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int idy = threadIdx.y + blockDim.y * blockIdx.y;
        int offsetx = blockDim.x * gridDim.x;
        int offsety = blockDim.y * gridDim.y;
        int i, j;
        uchar4 p,p1;
        double g12,g21,g22,g11;
        for(i = idx; i < w; i += offsetx)
                for(j = idy; j < h; j += offsety){
                        p1 = tex2D(tex, i, j);
                        g11 = RGBToWB(~p1.x,~p1.y,~p1.z);
                        if (i + 1 == w) {
                                g12 = g11;
                        }
                        else {
                                p = tex2D(tex, i+1, j);
                                g12=  RGBToWB(~p.x,~p.y,~p.z);
                        }
                        if (j + 1 == h) {
                                g21 = g11;
                        }
                        else {
                                p = tex2D(tex, i, j+1);
                                g21 = RGBToWB(~p.x,~p.y,~p.z);
                        }
                        if (j + 1 == h || i + 1 == w) {
                                if(j + 1 == h)
                                    g22 = g12;
                                else
                                    g22 = g21;
                        }
                        else {
                                p = tex2D(tex, i+1, j+1);
                                g22 = RGBToWB(~p.x,~p.y,~p.z);
                        }
                        g11 -= g22;
                        g21 -= g12;
                        g11 = f(g11*g11 + g21 * g21);
                        g11=chek(g11);
                        dst[j * w + i] = make_uchar4(g11,g11,g11, p1.w);
                }
}

int main() {
        int h, w;
        char path_in[257];
        char path_out[257];

        scanf("%s", path_in);
        scanf("%s", path_out);
        FILE* in = fopen(path_in, "rb");
        fread(&w, sizeof(int), 1, in);
        fread(&h, sizeof(int), 1, in);
        uchar4 *img = (uchar4 *)malloc(sizeof(uchar4) * h * w);
        fread(img, sizeof(uchar4), h * w, in);
        fclose(in);

        cudaArray *dev_arr;
        cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
        CSC(cudaMallocArray(&dev_arr, &ch, w, h));
        CSC(cudaMemcpyToArray(dev_arr, 0,
         0, img, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

        tex.addressMode[0] = cudaAddressModeClamp;
        tex.addressMode[1] = cudaAddressModeClamp;
        tex.channelDesc = ch;
        tex.filterMode = cudaFilterModePoint;
        tex.normalized = false;
        CSC(cudaBindTextureToArray(tex, dev_arr, ch));

        uchar4 *dev_img;
        CSC(cudaMalloc(&dev_img, sizeof(uchar4) * w * h));
        kernel<<< dim3(32, 32), dim3(32, 32) >>>(dev_img, w, h);
        CSC(cudaMemcpy(img, dev_img, sizeof(uchar4)
         * w * h, cudaMemcpyDeviceToHost));

        FILE* out = fopen(path_out, "wb");
        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);
        fwrite(img, sizeof(uchar4), h * w, out);
        fclose(out);
        CSC(cudaUnbindTexture(tex));
        CSC(cudaFreeArray(dev_arr));
        CSC(cudaFree(dev_img));
        free(img);
        return 0;
}
