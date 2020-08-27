#include <stdio.h>
#include "helper_cuda.h"
#include "GPGPU_HEAT.h"
/*fisier care contine apelurile CUDA*/
 /*se stabileste numarul de thread-uri/ block-uri*/
dim3 threadsPerBlock(32, 32);
dim3 numBlocks((M + 2) / threadsPerBlock.x, (N + 2) / threadsPerBlock.y);


/*functie care incadreaza o valoare data intr-un interval de valori*/
__device__ unsigned char clamp(int value, unsigned char min, unsigned char max) {
	if (value < min) return min;
	if (value > max) return max;
	return value;
}
/*functie implementata pe baza valorilor date in exemplul din cerinta pentru
transformarea din grade kelvin in valori RGB*/
__device__ RGBtemp kelvinToRGB(double kelvin) {
	RGBtemp result;
	int blue, red, green;
	double temp = (kelvin + 273) / 20;
	if (temp <= 66) {
		red = 255;
		green = 99.4708025861 * log(temp) - 161.1195681661;
		if (temp <= 19) {
			blue = 0;
		}
		else {
			blue = 138.5177312231 * log(temp - 10) - 305.0447927307;
		}
	}
	else {
		red = 329.698727446 * pow(temp - 60, -0.1332047592);
		green = 288.1221695283 * pow(temp - 60, -0.0755148492);
		blue = 255;
	}
	result.B = clamp(blue, 0, 255);
	result.G = clamp(green, 0, 255);
	result.R = clamp(red, 0, 255);
	return result;
}
/*functie de la nivelul GPU care calculeaza valoarea de la pasul curent pe baza
valorilor de la pasul precedent din punctul curent si din vecinatate*/
__global__ void computeTemp(double(*pa)[M + 2], double(*pb)[M + 2]) {
	int i, j, k, l;
	double sum = 0.0;
	i = (blockIdx.x * blockDim.x) + threadIdx.x;
	j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if ((i < M + 1) && (j < N + 1) && (i != 0) && (j != 0)) {
		for (k = i - 1; k <= i + 1; k++) {
			for (l = j - 1; l <= j + 1; l++) {
				sum = sum + pa[k][l];
			}
		}
		pb[i][j] = sum / 9.0;
	}
	if ((i == M + 1) || (i == 0) || (j == N + 1) || (j == 0)) {
		pb[i][j] = tr;
	}
	if ((i == x0) && (j == y0)) {
		pb[i][j] = ts;
	}
}
/*functie de la nivelul GPU care calculeaza diferenta de temperatura dintre ultimii pasi
in fiecare punct din simulare; se actualizeaza valoarea lui pa cu valorile nou calculate din pb
in cadrul functiei computeTemp*/
__global__ void computeDiff(double(*pa)[M + 2], double(*pb)[M + 2], double(*pdiff)[M + 2]) {
	int i, j;
	i = (blockIdx.x * blockDim.x) + threadIdx.x;
	j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if ((i < M + 2) && (j < N + 2)) {
		pdiff[i][j] = abs(pb[i][j] - pa[i][j]);
		pa[i][j] = pb[i][j];
	}
}
/*functie de la nivelul GPU care transforma pentru fiecare punct al simularii valoarea stocata
din grade kelvin in valori RGB*/
__global__ void computeRGBfromKelvin(double(*pa)[M + 2], RGBtemp(*prgb)[M + 2]) {
	int i, j;
	i = (blockIdx.x * blockDim.x) + threadIdx.x;
	j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if ((i < M + 2) && (j < N + 2)) {
		prgb[i][j] = kelvinToRGB(pa[i][j]);
	}
}
/*apel care rezolva problema identificarii corecte a structurii <<< >>> 
din fisierul cpp pentru functia computeTemp*/
void runComputeTemp(double(*pa)[M + 2], double(*pb)[M + 2]) {
	computeTemp<<<numBlocks,threadsPerBlock>>>(pa, pb);
}
/*apel care rezolva problema identificarii corecte a structurii <<< >>>
din fisierul cpp pentru functia computeDiff*/
void runComputeDiff(double(*pa)[M + 2], double(*pb)[M + 2], double(*pdiff)[M + 2]) {
	computeDiff<<<numBlocks,threadsPerBlock>>>(pa, pb, pdiff);
}
/*apel care rezolva problema identificarii corecte a structurii <<< >>>
din fisierul cpp pentru functia computeRGBfromKelvin*/
void runComputeRGBfromKelvin(double(*pa)[M + 2], RGBtemp(*prgb)[M + 2]) {
	computeRGBfromKelvin<<<numBlocks,threadsPerBlock>>>(pa, prgb);
}
