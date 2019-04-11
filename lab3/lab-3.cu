#include <stdio.h>
#include <stdlib.h>
#include <cmath>

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


//texture<uchar4, 2, cudaReadModeElementType> tex;
__constant__ double3 conTest[33];
__global__ void kernel(uchar4 *dst, int w, int h, int n) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int idy = threadIdx.y + blockDim.y * blockIdx.y;
        int offsetx = blockDim.x * gridDim.x;
        int offsety = blockDim.y * gridDim.y;
        int i, j,k;
        uchar4 a;
        double3 b;
        int ans;
        double max,tmp;
        for(i = idx; i < w; i += offsetx){
                for(j = idy; j < h; j += offsety){
                        max=1000000;
                        a = dst[j * w + i];
                        for(k=0;k<n;k++){
                            b=conTest[k];
                            tmp=SQRT(POW(a.x-b.x)+
                            POW(a.y-b.y)+
                            POW(a.z-b.z));
                            //zzz1(b);
                            //printf(" %d %d %f %f\n",j * w + i,k,tmp,max);
                            if(max>tmp){
                                max=tmp;
                                ans=k;
                            }
                        }
                        dst[j * w + i] = make_uchar4(a.x,a.y,a.z, ans);
                }
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
        int n,x,y;
        scanf("%d",&n);
        double3 *test = (double3 *)malloc(sizeof(double3) * n);
        double3 *testNext = (double3 *)malloc(sizeof(double3) * n);
        double4 *res = (double4 *)malloc(sizeof(double4) * n);
        for(int i=0;i<n;i++){
            scanf("%d%d",&x,&y);

            test[i].x=img[y*w+x].x;
            test[i].y=img[y*w+x].y;
            test[i].z=img[y*w+x].z;
            /*zzz1(test[i]);
            printf("\n");*/
        }
        /*cudaArray *dev_arr;
        cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
        CSC(cudaMallocArray(&dev_arr, &ch, w, h));
        CSC(cudaMemcpyToArray(dev_arr, 0,
         0, img, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));



        tex.addressMode[0] = cudaAddressModeClamp;
        tex.addressMode[1] = cudaAddressModeClamp;
        tex.channelDesc = ch;
        tex.filterMode = cudaFilterModePoint;
        tex.normalized = false;
        CSC(cudaBindTextureToArray(tex, dev_arr, ch));*/

        uchar4 *dev_img;
        CSC(cudaMalloc(&dev_img, sizeof(uchar4) * w * h));
            CSC(cudaMemcpy ( dev_img, img, sizeof(uchar4)
         * w * h, cudaMemcpyHostToDevice ));
        y=1;
         while (y) {
            /*for(int i=0;i<n;i++){
                zzz1(test[i]);
            }*/
            CSC( cudaMemcpyToSymbol(conTest, test, n*sizeof(double3)) );
            kernel<<< dim3(32, 32), dim3(32, 32) >>>(dev_img, w, h,n);
            CSC(cudaMemcpy(img, dev_img, sizeof(uchar4)
         * w * h, cudaMemcpyDeviceToHost));

            for(int i=0;i<n;i++){
                res[i].x = res[i].y = res[i].z = res[i].w = 0;
            }
            for(int i = 0; i < w*h; i++){
                    //zzz3(img[i]);
                    //printf("%d %d \n",i,img[i].w);
                    res[img[i].w].x+=img[i].x;
                    res[img[i].w].y+=img[i].y;
                    res[img[i].w].z+=img[i].z;
                    res[img[i].w].w+=1;
                    //printf("%f %f     ",res[img[i].w].x,res[img[i].w].w);
            }
            //printf("\n");
            for(int i=0;i<n;i++){
                testNext[i].x=res[i].x/res[i].w;
                testNext[i].y=res[i].y/res[i].w;
                testNext[i].z=res[i].z/res[i].w;
                //zzz1(testNext[i]);
            }
            //printf("NEXT\n");
            y=0;
            for(int i=0;i<n;i++){
                if(test[i].x!=testNext[i].x){
                    y=1;
                }
                if(test[i].y!=testNext[i].y){
                    y=1;
                }
                if(test[i].z!=testNext[i].z){
                    y=1;
                }
            }
            for(int i=0;i<n;i++){
                test[i].x=testNext[i].x;
                test[i].y=testNext[i].y;
                test[i].z=testNext[i].z;
            }

        }


        FILE* out = fopen(path_out, "wb");
        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);
        fwrite(img, sizeof(uchar4), h * w, out);
        fclose(out);
        CSC(cudaFree(dev_img));
        free(img);
        free(test);
         free(testNext);
         free(res);
        return 0;
}
