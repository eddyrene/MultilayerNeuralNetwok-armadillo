#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
using namespace std;

//#define THREADS_PER_BLOCK 32

void fillMatrix(int* a, int n)
{
   int i;
   for (i = 0; i < n*n; ++i)
        a[i] = 10;//rand()%5;
}

__global__ 
void matrixAdition(int *c, int *a, int *b,int n) 
{
    int ij = threadIdx.x + blockDim.x * blockIdx.x;
		if(ij<(n*n))
			c[ij] = a[ij] + b[ij];
}

__global__ 
void matrixAditionRow(int *c, int *a, int *b,int n) 
{
   	int ij = threadIdx.x + blockDim.x * blockIdx.x;
   //	if(blockDim.x != 0)
   	//printf("%d  salida\n", ij);
	for(int i =0 ;i<n;i++)
	{
		if(ij<n)
			c[ij*n+i] = a[ij*n+i] + b[ij*n+i];
	}
}

__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P,int Mask_Width , int Width)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	float Pvalue = 0;
	int N_start_point = i - (Mask_Width/2);
	for (int j = 0; j < Mask_Width; j++) 
	{
		if (N_start_point + j >= 0 && N_start_point + j < Width) 
		{
			Pvalue += N[N_start_point + j]*M[j];
		}
	}
	P[i] = Pvalue;
}

__global__ 
void matrixAditionCol(int *c, int *a, int *b,int n) 
{
   	int ij = threadIdx.x + blockDim.x * blockIdx.x;
	for(int i =0 ;i<n;i++)
	{
		if(ij<n)
			c[ij+n*i] = a[ij+n*i] + b[ij+n*i];
	}
}

void printMatrix(string s, int *a , int tam){
	cout<<s;
	for(int i=0;i<tam;i++)
	{
		for(int j=0;j<tam;j++)
		{
			cout<<a[i*tam+j]<<" ";
		}
		cout<<endl;
	}
}

void ReadPPM(int *Pin, char *name)
{
	int e1;
    string line,s1;
    ifstream file(name);
    getline(file,line);
    getline(file,line);
    getline(file,line);
    getline(file,line);
    int m=0;
    while(!file.eof())
    {
        file>>e1;
       // cout<<e1<<endl;
        //if(!e1) break;
        Pin[m]=e1;
        m++;
    }
}

void WritePGM(int * Pout, int fil , int cols, char *name)
{
    ofstream file(name);
    file<<"P2"<<endl;
    file<<"# Function ConvertRGBtoGray @eddyrene"<<endl;
    file<<fil<<" "<<cols<<endl;
    file<<255<<endl;
    int n = fil*cols;
    int i=0;
    while(i<n)
    {
        file<<Pout[i]<<endl;
        i++;
    }
}

void WritePPM(int * Pout, int fil , int cols, char *name)
{
    ofstream file(name);
    file<<"P3"<<endl;
    file<<"# Function ConvertRGBtoGray @eddyrene"<<endl;
    file<<fil<<" "<<cols<<endl;
    file<<255<<endl;
    int n = fil*cols;
    int i=0;
    while(i<3*n)
    {
        file<<Pout[i]<<endl;
        i++;
    }
}

int main(int argc, char *argv[])
{
	srand (time(NULL));
	int  N= strtol(argv[1], NULL, 10);
		//matrixAditionCol<<<blocks2,THREADS_PER_BLOCK>>>( d_c, d_a, d_b,N);
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start,stop);
	printf("Elapsed time : %f ms\n" ,elapsedTime);
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	//printMatrix("Printing Matrix A \n",a,N);
	//printMatrix("Printing Matrix B \n",b,N);
	//printMatrix("Printing Matrix C \n",c,N);
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}
