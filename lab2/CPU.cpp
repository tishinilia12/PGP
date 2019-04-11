#include <iostream>
#include <iomanip>
#include <cctype>
#include <string>
#include <vector>
#include <cmath>
#include <set>
#include <queue>
#include <iterator>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <stack>
#include <fstream>
#include <bitset>
#include <ctime>
#include <stdio.h>
//#include <unordered_map>

using namespace std;

#define pb push_back
#define mp make_pair
#define mt make_tuple
#define ft first
#define sd second
#define all(x) (x).begin(),(x).end()
#define mfor(i,b,e) for(int i=b;i<e;i++)
#pragma warning(disable : 4996)

#define RGBToWB(r, g, b)  0.299*r + 0.587*g + 0.114*b;

/*    texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4 *dst, int w, int h) {
int idx = threadIdx.x + blockDim.x * blockIdx.x;
int idy = threadIdx.y + blockDim.y * blockIdx.y;
int offsetx = blockDim.x * gridDim.x;
int offsety = blockDim.y * gridDim.y;
int i, j,i1,j1;
uchar4 p;
for(i = idx; i < w; i += offsetx)
for(j = idy; j < h; j += offsety){

p = tex2D(tex, i, j);
dst[j * w + i] = make_uchar4(~p.x, ~p.y, ~p.z, p.w);
}
}*/

int main() {
	int h, w;

	char path_in[257];
	char path_out[257];

	scanf("%s", path_in);

	FILE *in = fopen(path_in, "rb");
	fread(&w, sizeof(int), 1, in);
	fread(&h, sizeof(int), 1, in);
	char *img = (char *)malloc(sizeof(char) * h * w * 4);
	char *omg = (char *)malloc(sizeof(char) * h * w * 4);
	fread(img, sizeof(char), h * w * 4, in);
	fclose(in);



	for (int i = 0; i<w; i++) {
		for (int j = 0; j<h; j++) {
			long double g11 = RGBToWB(img[j*w * 4 + i * 4],
				img[j*w * 4 + i * 4 + 1], img[j*w * 4 + i * 4 + 2]);
			long double g12,g21,g22;
			if (i + 1 == w) {
				g12 = g11;
			}
			else {
				g12=  RGBToWB(img[j*w * 4 + i * 4+4],
					img[j*w * 4 + i * 4 + 1+4], img[j*w * 4 + i * 4 + 2+4]);
			}
			if (j + 1 == h) {
				g21 = g11;
			}
			else {
				g21 = RGBToWB(img[j*w * 4 + i * 4 + 4*w],
					img[j*w * 4 + i * 4 + 1 + 4*w], img[j*w * 4 + i * 4 + 2 + 4*w]);
			}
			if (j + 1 == h || i + 1 == w) {
				if(j + 1 == h)
					g22 = g12;
				else
					g22 = g21;
			}
			else {
				g22 = RGBToWB(img[j*w * 4 + i * 4 + 4 * w+4],
					img[j*w * 4 + i * 4 + 1 + 4 * w+4], img[j*w * 4 + i * 4 + 2 + 4 * w+4]);
			}
			g11 -= g22;
			g21 -= g12;
			g11 = sqrt(g11*g11 + g21 * g21);
			omg[j*w * 4 + i * 4 + 3] = img[j*w * 4 + i * 4 + 3];
			omg[j*w * 4 + i * 4] = g11;
			omg[j*w * 4 + i * 4+1] = g11;
			omg[j*w * 4 + i * 4+2] = g11;
		}
	}

	scanf("%s", path_out);
	FILE *out = fopen(path_out, "wb");
	fwrite(&w, sizeof(int), 1, out);
	fwrite(&h, sizeof(int), 1, out);
	fwrite(omg, sizeof(char), 4 * h * w, out);
	fclose(out);
	free(img);

	return 0;
}
