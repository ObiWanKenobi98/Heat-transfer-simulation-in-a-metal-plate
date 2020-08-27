// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include "device_launch_parameters.h"

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <time.h>
#include <windows.h>

#include "GPGPU_HEAT.h"

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#else
#define KERNEL_ARGS2(grid, block)
#endif
/*fisier care contine implementarea propriu-zisa a cerintelor*/
/*bufferul in care se memoreaza pixelii*/
GLubyte* PixelBuffer = new GLubyte[WIDTH * HEIGHT * 3];
/*indicator care semnaleaza modificarea parametrilor de intrare*/
bool newParameters;
/*functia de afisare pe ecran*/
void display()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, PixelBuffer);
	glutSwapBuffers();
}
/*thread care se ocupa de afisarea imaginii corespunzatoare temperaturii
pentru placa simulata(glutMainLoop() este un apel blocant, nu am putut
sa il apelez direct in main)*/
DWORD WINAPI ThreadFunc(void* data) {
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(400, 400);
	glutInitWindowPosition(100, 100);
	int MainWindow = glutCreateWindow("Heat Transfer Simulation");
	glClearColor(0.0, 0.0, 0.0, 0);
	glutDisplayFunc(display);
	glutMainLoop();
	return 0;
}
/*schelet pentru subpunctul de load balancing*/
DWORD WINAPI ComputeTemperatureCPU(void* data) {
	myCPUdata* localData;
	localData = (myCPUdata*)malloc(sizeof(myCPUdata));
	if (!localData) return -1;
	memcpy(localData, data, sizeof(myCPUdata));
	free(localData);
	return 0;
}
/*schelet pentru subpunctul de modificare a parametrilor in timp real*/
DWORD WINAPI ModifyParameters(void* data) {
	int localx0, localy0;
	double localts, localtr;
	double localLoadFactor;
	scanf("%d", &localx0);
	scanf("%d", &localy0);
	scanf("%lf", &localts);
	scanf("%lf", &localtr);
	scanf("%lf", &localLoadFactor);
	newParameters = true;
	return 0;
}
/*calculul efectiv al simularii; se instantiaza thread-urile de la nivelul GPU, se efectueaza schimb de date
cu CPU la fiecare pas pentru a vedea daca s-a indeplinit conditia de oprire a simularii, se afiseaza valorile
termice pe ecran; se reiau pasii anteriori daca simularea trebuie sa continue; gestiune de memorie device/host;
introducerea datelor de intrare are loc de la tastatura*/
int main(int argc, char* argv[]) {
	newParameters = false;
	/*structuri host necesare calculului: a->valorile de la pasul curent, b->valorile de la pasul urmator,
	diff->diferenta de temperatura dintre 2 pasi succesivi, RGBtemp->valorile RGB asociate unei temperaturi
	kelvin la un anumit pas*/
	double a[M + 2][N + 2];
	double b[M + 2][N + 2];
	double diff[M + 2][N + 2];
	RGBtemp RGBtemperature[M + 2][N + 2];
	/*se initializeaza glut pentru fereastra cu valorile termice ale simularii*/
	glutInit(&argc, argv);
	/*thread-uri pentru gestiunea parametrilor de intrare si a afisarii ferestrei pe ecran*/
	HANDLE thread = CreateThread(NULL, 0, ThreadFunc, NULL, 0, NULL);
	HANDLE inputThread = CreateThread(NULL, 0, ModifyParameters, NULL, 0, NULL);
	/*structuri device necesare calcului, corespondentele celor de mai sus*/
	double(*pa)[M + 2];
	double(*pb)[M + 2];
	double(*pdiff)[M + 2];
	RGBtemp(*prgb)[M + 2];
	int i, j;
	/*se aloca memorie pentru structurile device*/
	cudaMalloc((void**)&pa, (N + 2) * (M + 2) * sizeof(double));
	cudaMalloc((void**)&pb, (N + 2) * (M + 2) * sizeof(double));
	cudaMalloc((void**)&pdiff, (N + 2) * (M + 2) * sizeof(double));
	cudaMalloc((void**)&prgb, (N + 2) * (M + 2) * sizeof(RGBtemp));
	/*se initializeaza structurile host*/
	for (i = 0; i < M + 2; i++) {
		for (j = 0; j < N + 2; j++) {
			a[i][j] = tr;
			b[i][j] = tr;
			diff[i][j] = 0.0;
			RGBtemperature[i][j].R = 0;
			RGBtemperature[i][j].G = 0;
			RGBtemperature[i][j].B = 0;
		}
	}
	a[x0][y0] = ts;
	b[x0][y0] = ts;
	/*se initializeaza structurile device cu valorile din host*/
	cudaMemcpy(pa, a, (N + 2) * (M + 2) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(pb, b, (N + 2) * (M + 2) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(pdiff, diff, (N + 2) * (M + 2) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(prgb, RGBtemperature, (N + 2) * (M + 2) * sizeof(RGBtemp), cudaMemcpyHostToDevice);
	double sum = EPS + 1;
	/*simularea propriu-zisa are loc in aceasta bucla*/
	while (sum > EPS) {
		clock_t start_t, end_t;
		double total_t;
		start_t = clock();
		if (newParameters == true) {
			CloseHandle(inputThread);
			inputThread = CreateThread(NULL, 0, ModifyParameters, NULL, 0, NULL);
		}
		/*se aplica, pe rand, fiecare functie device asupra datelor de la pasul curent*/
		runComputeTemp(pa, pb);
		cudaDeviceSynchronize();
		runComputeDiff(pa, pb, pdiff);
		cudaDeviceSynchronize();
		runComputeRGBfromKelvin(pa, prgb);
		cudaDeviceSynchronize();
		/*se actualizeaza in memoria host datele calculate pentru urmatorul pas de device*/
		cudaMemcpy(diff, pdiff, (N + 2) * (M + 2) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(a, pa, (N + 2) * (M + 2) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(RGBtemperature, prgb, (N + 2) * (M + 2) * sizeof(RGBtemp), cudaMemcpyDeviceToHost);
		/*se verifica conditia de convergenta*/
		sum = 0.0;
		for (i = 0; i < M + 2; i++) {
			for (j = 0; j < N + 2; j++) {
				sum = sum + diff[i][j];
			}
		}
		for (i = 0; i < M + 2; i++) {
			for (j = 0; j < N + 2; j++) {
				PixelBuffer[3 * (i * (M + 2) + j)] = RGBtemperature[i][j].R;
				PixelBuffer[3 * (i * (M + 2) + j) + 1] = RGBtemperature[i][j].G;
				PixelBuffer[3 * (i * (M + 2) + j) + 2] = RGBtemperature[i][j].B;
			}
		}
		end_t = clock();
		/*optional->timpul trecut in cadrul unui pas al simularii; precizie slaba*/
		total_t = double(end_t - start_t) / double(CLOCKS_PER_SEC);
		//printf("%lf\n", total_t);
	}
	printf("Convergence reached!\n");
	/*se elibereaza memoria pentru structurile device, se inchid handle-urile corespunzatoare
	thread-urilor deschise anterior*/
	Sleep(50000);
	CloseHandle(thread);
	CloseHandle(inputThread);
	cudaFree(pa);
	cudaFree(pb);
	cudaFree(pdiff);
	cudaFree(prgb);
	return 0;
}