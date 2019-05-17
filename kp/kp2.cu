#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <unistd.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <iostream>
using namespace std;
using namespace thrust; 

#define CSC(call) {														\
	 cudaError err = call;												\
	 if (err != cudaSuccess) {											\
		  fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
				__FILE__, __LINE__, cudaGetErrorString(err));			\
		  exit(1);														\
	 }																	\
} while (0)








const int PTS_NUM = 3000;
const int PTS_MULT = 70;
const float INF = FLT_MAX - 10;
const int W = 1024;
const int H = 648;

#define MAX(a,b) a>b?a:b
#define MIN(a,b) a<b?a:b

struct Point {
	float x,y;
    float mx, my;
    float vx, vy;
};
  
 struct OBCmp1 {
  __host__ __device__
  bool operator()(const Point& a, const Point& b) {
  		float t1=a.y, t2=b.y;
  		if(t1<0){
  			t1-=0.4;
  		}
  		if(t2<0){
  			t2-=0.4;
  		}
  		if(((int)(t1/0.4))<((int)(t2/0.4))){
  			return 1;
  		}
      	return a.x < b.x;
  }
};
struct OBCmp2 {
  __host__ __device__
  bool operator()(const Point& a, const Point& b) {
      float t1=a.x, t2=b.x;
  		if(t1<0){
  			t1-=0.4;
  		}
  		if(t2<0){
  			t2-=0.4;
  		}
  		if(((int)(t1/0.4))<((int)(t2/0.4))){
  			return 1;
  		}
      	return a.y < b.y;
  }
};
  bool changed=0;
    float2 mass=make_float2(0.0,0.0);
float *ptsx, *ptsy; 
Point *points;
curandState *gen;  
float2 *gm;  

GLuint vbo;
struct cudaGraphicsResource *txtBuffer;


__device__ __host__ float fun1(float x, float y) {
	return -(abs(x) * sinf(sqrtf(abs(x)))) -(abs(y) * sinf(sqrtf(abs(y))));
	
}


__device__ __host__ float fun(int i, int j,float xc,float sx,float yc,float sy) {
	float x = 2.0 * i / (float)(W - 1) - 1.0;
	float y = 2.0 * j / (float)(H - 1) - 1.0;
	return fun1(xc + sx * x, yc + sy * y);
}
float w=0.99, a1=0.8, a2=0.2, dt=0.01, G=0.01;
__device__ uchar4 get_color(float f) {
	float k = 1.0 / 5.0;
	if (f <  k)
		return make_uchar4(255, (int)((f - k) * 255 / k), 0, 0);
	if (f < 2 * k)
		return make_uchar4(255, 255, (int)((f - 2 * k) * 255 / k), 0);
	if (f < 3 * k)
		return make_uchar4(255 - (int)((f - 3 * k) * 255 / k), 255, 255, 0);
	if (f < 4 * k)
		return make_uchar4(0, 255 - (int)((f - 4 * k) * 255 / k), 255, 0);
	if (f < 5 * k)
		return make_uchar4(0, 0, 255 - (int)((f - 5 * k) * 255 / k), 0);
	return make_uchar4(0, 0, 0, 0);
}

__global__ void fillMap(float *map,float xc,float sx,float yc,float sy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;

    for (j = idy; j < H; j += offsety)
        for (i = idx; i < W; i += offsetx)
            map[j * W + i] = fun(i, j,xc,sx,yc,sy);
}

 float xc=0, yc=0, sx=18, sy=18.0*H/float(W), minf=INF, maxf=-INF;
void searchExtremes(float *map) {
    device_ptr<float> p_arr = device_pointer_cast(map);
    maxf= *max_element(p_arr, p_arr + W * H);
    minf= *min_element(p_arr, p_arr + W * H);
}


__global__ void float2uchar(uchar4 *data, float *map,float minf,float maxf) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;
	float f;

    for (j = idy; j < H; j += offsety)
	    for (i = idx; i < W; i += offsetx) {
            f = (map[j * W + i] - minf) / (maxf - minf);
            data[j * W + i] = get_color(f);
		}
}


float2 calculateCenterOfMass() {
    device_ptr<float> x_arr = device_pointer_cast(ptsx);
    device_ptr<float> y_arr = device_pointer_cast(ptsy);
    double xsum = reduce(x_arr, x_arr + PTS_NUM);
    double ysum = 
    reduce(y_arr, y_arr + PTS_NUM);

    float2 masstmp;
    masstmp.x = xsum / (float)PTS_NUM;
    masstmp.y = ysum / (float)PTS_NUM;

    return (abs(mass.x - masstmp.x) > 1 ||
            abs(mass.y -masstmp.y) > 1 ) ? masstmp : mass;
}


void shiftWindow(float2 masstmp) {
    if (mass.x > masstmp.x)
        xc -= mass.x - masstmp.x;
    else if (mass.x < masstmp.x)
        xc += masstmp.x - mass.x;
    if (mass.y > masstmp.y)
        yc -= mass.y - masstmp.y;
    else if (mass.y < masstmp.y)
        yc += masstmp.y- mass.y;
}


__global__ void updatePoints (Point *points, float2 *force, float2 *gm,
  float *ptsx, float *ptsy, time_t seed, curandState *gen,
  float w,float a1,float a2,float dt,float G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int i;
    float r1, r2;

    curand_init(threadIdx.x, 0, 0, gen);

    for (i = idx; i < PTS_NUM; i += offsetx) {

        r1 = curand_uniform(gen);
        r2 = curand_uniform(gen);

        points[i].vx = w * points[i].vx + \
                       (a1 * r1 * (points[i].mx - points[i].x) + \
                        a2 * r2 * (gm->x - points[i].x) + \
                        G * force[i].x) * dt;
        points[i].vy = w * points[i].vy + \
                       (a1 * r1 * (points[i].my - points[i].y) + \
                        a2 * r2 * (gm->y - points[i].y) + \
                        G * force[i].y) * dt;

        points[i].x = points[i].x + points[i].vx * dt;
        points[i].y = points[i].y + points[i].vy * dt;
        ptsx[i]=points[i].x;
        ptsx[i]=points[i].y;
        if (fun1(points[i].x, points[i].y) < fun1(points[i].mx, points[i].my)) {
            points[i].mx = points[i].x;
            points[i].my = points[i].y;
        }
    }
}

__global__ void nullForce(float2 *F){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int i;
	for (i = idx; i < PTS_NUM; i += offsetx) {
		F[i].x=0;
		F[i].y=0;
	}
}
__global__ void calcForce(float2 *F, Point *points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int i, j;

    float dist;
    for (i = idx; i < PTS_NUM; i += offsetx) {
        for (j = MAX(i-10,0); j < MIN(PTS_NUM,i+10); ++j) {
            if (j == i) continue;
            dist = __powf(sqrtf((points[i].x - points[j].x) * (points[i].x - points[j].x) + \
                           (points[i].y - points[j].y) * (points[i].y - points[j].y)), 3.0);
            F[i].x += (points[i].x - points[j].x) / (dist+0.000001);
            F[i].y += (points[i].y - points[j].y) / (dist+0.000001);
        }
    }
}


__global__ void updateGlobalMinimum (Point *points, 
float2 *gm) {
    for (int i = 0; i < PTS_NUM; ++i)
        if (fun1(points[i].mx, points[i].my) < fun1(gm->x, gm->y)) {
            gm->x = points[i].mx;
            gm->y = points[i].my;
        }
}


__global__ void renderPoints (Point *points, uchar4 *data, float *ptsx, float 
    *ptsy, bool paintOver,float xc,float sx,float yc,float sy,float minf,float maxf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int k;
    float x, y;

    for (k = idx; k < PTS_NUM; k += offsetx) {
        x = points[k].x;
        y = points[k].y;

        if ((x > (xc - sx) && x < (xc + sx)) && \
            (y > (yc - sy) && y < (yc + sy))) {
            int i = (int)((x - xc + sx) / (2 * sx / (float)(W - 1)));
            int j = (int)((y - yc + sy) / (2 * sy / (float)(H - 1)));

            if (paintOver) {
                float func = (fun(i, j,xc,sx,yc,sy) - minf) / (maxf - minf);
                data[j * W + i] =get_color(func);
            } else {
                data[j * W + i] =get_color(7.0);
            }

            
        }
    }
}


void update() {
	uchar4 *txt;
	size_t size;
	CSC( cudaGraphicsMapResources(1, &txtBuffer, 0) );
	CSC( cudaGraphicsResourceGetMappedPointer((void**)&txt, &size, txtBuffer) );

    float2 masstmp = calculateCenterOfMass();

    if ((masstmp.x != mass.x || masstmp.y != mass.y) || changed) {
        shiftWindow(masstmp);
        mass = masstmp;

        float *map;
        CSC( cudaMalloc(&map, W * H * sizeof(float)) );

        fillMap <<<dim3(32, 32), dim3(8, 32)>>> (map,xc,sx,yc,sy);
        CSC( cudaGetLastError() );

        searchExtremes(map);

        float2uchar <<<dim3(32, 32), dim3(8, 32)>>> (txt, map,minf,maxf);
        CSC( cudaGetLastError() );

        CSC( cudaFree(map) );

        changed = false;

    } else {
        // Paint over old points
        renderPoints <<<64, 64>>> (points, txt, ptsx, ptsy, true
        , xc, sx, yc, sy, minf, maxf);
        CSC( cudaGetLastError() );
    }

    float2 *force;
    CSC( cudaMalloc(&force, sizeof(float2) * PTS_NUM) );
    nullForce<<<64,64>>>(force);

    sort(points,points+PTS_NUM,OBCmp1());
    calcForce <<<64, 64>>> (force,points);
    sort(points,points+PTS_NUM,OBCmp2());
    calcForce <<<64, 64>>> (force,points);

    CSC( cudaGetLastError() );

    updatePoints <<<64, 64>>> (points, force, gm, ptsx, ptsy, time(NULL), gen,
      w, a1, a2, dt, G);
    CSC( cudaGetLastError() );

    CSC( cudaFree(force) );

    updateGlobalMinimum <<<1, 1>>> (points, gm);
    CSC( cudaGetLastError() );

    renderPoints <<<64, 64>>> (points, txt, ptsx, ptsy, false
    ,xc, sx, yc, sy, minf, maxf);
    CSC( cudaGetLastError() );

    CSC( cudaGraphicsUnmapResources(1, &txtBuffer, 0) );

    glutPostRedisplay();
}


void display() {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(W, H, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}


void keys(unsigned char key, int x, int y) {
	if (key == 27) {
		CSC( cudaGraphicsUnregisterResource(txtBuffer) );
		glBindBuffer(1, vbo);
		glDeleteBuffers(1, &vbo);
		exit(0);
	}
    else if (key == 113){  // q
         a1 -= 0.05;
         cout << "1: " << a1 <<endl;
        }
    else if (key == 97){  // a
        a1 += 0.05;
        cout << "1: " << a1 <<endl;
        }
    else if (key == 119){  // w
        a2 -= 0.05;
        cout << "2: " << a2 << endl;
        }
    else if (key == 115){  // s
           a2 += 0.05;
           cout << "2: " << a2 << endl;
        }
    else if (key == 101){  // e
        dt -= 0.01;
        cout << "dt: " << dt << endl;
                        }
    else if (key == 100){  // d
        dt += 0.01;
        cout << "dt: " << dt << endl;
        }
    else if (key == 114){  // r
        w -= 0.02;
        cout << "w: " << w << endl;
        }
    else if (key == 102){  // f
        w += 0.02;
        cout << "w: " << w << endl;
        }
}


void processSpecialKeys(int key, int x, int y) {
	switch(key) {
		case GLUT_KEY_HOME:
			sx *= 0.97;
			sy = sx * (float)H / (float)W;
			break;
		case GLUT_KEY_END:
			sx *= 1.03;
			sy = sx * (float)H / (float)W;
			break;
		case GLUT_KEY_LEFT:
			xc -= 0.01*sx;
			break;
		case GLUT_KEY_RIGHT:
			xc += 0.01*sx;
			break;
		case GLUT_KEY_DOWN:
			yc -= 0.01*sy;
			break;
		case GLUT_KEY_UP:
			yc += 0.01*sy;
			break;
	}
    changed = true;
}


void generatePointsPositions() {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

     curandSetPseudoRandomGeneratorSeed(gen, time (NULL));
     curandGenerateUniform(gen, ptsx, PTS_NUM);

     curandSetPseudoRandomGeneratorSeed(gen, time (NULL));
     curandGenerateUniform(gen, ptsy, PTS_NUM);
}


__global__ void initPoints(Point *points, float *ptsx, float *ptsy, 
float2 *gm,float xc,float sx,float yc,float sy) {
    *gm = make_float2(0.0, 0.0);
    for (int i = 0; i < PTS_NUM; ++i) {
        // Map point position from (0, 1] to (-PTS_MULT / 2, PTS_MULT / 2]
        points[i].x = ptsx[i] * PTS_MULT - PTS_MULT / 2;
        points[i].y = ptsy[i] * PTS_MULT - PTS_MULT / 2;

        points[i].mx = points[i].x;
        points[i].my = points[i].y;
        points[i].vx = 0.0;
        points[i].vy = 0.0;

        if (fun1(points[i].x, points[i].y) < fun1(gm->x, gm->y)) {
            gm->x = points[i].x;
            gm->y = points[i].y;
        }
    }
}


int main(int argc, char **argv) {

    cout << time(NULL) << endl << flush;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(W, H);
	glutCreateWindow("Himmelblau");

	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(keys);
	glutSpecialFunc(processSpecialKeys);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLfloat)W, 0.0, (GLfloat)H);
	glewInit();

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, W * H * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);

	CSC( cudaGraphicsGLRegisterBuffer(&txtBuffer, vbo, cudaGraphicsMapFlagsWriteDiscard) );

    CSC( cudaMalloc(&ptsx, sizeof(float) * PTS_NUM) );
    CSC( cudaMalloc(&ptsy, sizeof(float) * PTS_NUM) );
    CSC( cudaMalloc(&points, sizeof(Point) * PTS_NUM) );
    CSC( cudaMalloc(&gen, sizeof(curandState)) );
    CSC( cudaMalloc(&gm, sizeof(float2)) );

    generatePointsPositions();
    initPoints <<<1, 1>>> (points, ptsx, ptsy, gm,xc, sx, yc, sy);

	glutMainLoop();

    CSC( cudaFree(ptsx) );
    CSC( cudaFree(ptsy) );
    CSC( cudaFree(points) );
}
