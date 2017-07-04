#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
using namespace std;

//#define THREADS_PER_BLOCK 32

#define Mask_width  3
#define Mask_radius Mask_width/2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0), 255))



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

__global__ 
void convolution_1D_basic_kernel(int *R, int *G, int *B , int *M, int *sd_R, int *sd_G, int *sd_B, int Mask_Width , int Width)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int r = 0;
	int g = 0;
	int b = 0;
	int N_start_point = i - (Mask_Width/2);
	for (int j = 0; j < Mask_Width; j++) 
	{
		if (N_start_point + j >= 0 && N_start_point + j < Width) 
		{
			r += R[N_start_point + j]*M[j];
			g += G[N_start_point + j]*M[j];
			b += B[N_start_point + j]*M[j];
		}
	}
	sd_R[i] = r;
	sd_G[i] = g;
	sd_B[i] = b;
}

__global__ 
void convolution(int *I, const int* __restrict__ M, int *P, int channels, int width, int height) 
{
   __shared__ int N_ds[w][w];
   int k;
   for (k = 0; k < channels; k++) {
      // First batch loading
      int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
         destY = dest / w, destX = dest % w,
         srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius,
         srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius,
         src = (srcY * width + srcX) * channels + k;
      if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
         N_ds[destY][destX] = I[src];
      else
         N_ds[destY][destX] = 0;

      // Second batch loading
      dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
      destY = dest / w, destX = dest % w;
      srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
      srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
      src = (srcY * width + srcX) * channels + k;
      if (destY < w) {
         if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = I[src];
         else
            N_ds[destY][destX] = 0;
      }
      __syncthreads();

      int accum = 0;
      int y, x;
      for (y = 0; y < Mask_width; y++)
         for (x = 0; x < Mask_width; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * Mask_width + x];
      y = blockIdx.y * TILE_WIDTH + threadIdx.y;
      x = blockIdx.x * TILE_WIDTH + threadIdx.x;
      if (y < height && x < width)
         P[(y * width + x) * channels + k] = clamp(accum);
      __syncthreads();
   }
}
/*
#define P2D(PTR, PITCH, ROW, COL, TYPE)    ((TYPE *)( (char *)(PTR) + (ROW) * (PITCH) ) )[(COL)]

__global__ void convolutionKernel(int *inImg, int *outImg, ROI size, int pitch, int *mask, size_t maskPitch, int maskSize, int maskSum)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row > size.height || col > size.width)
        return;

    int k = maskSize / 2;
    
    int pixelNewValue = 0;
    int p;
    int m;

    // Convolution
    for (int i = -k; i <= k; i++)
    {
        for (int j = -k; j <= k; j++)
        {
            // Pixel
            p = P2D(inImg, pitch, (row + i), (col + j), int);
            
            // Mask
            m = P2D(mask, maskPitch, (i+k), (j+k), int);
            
            pixelNewValue += m * p;
        }
    }

    // New value
    pixelNewValue /= maskSum;
    if (pixelNewValue < 0) pixelNewValue = 0;
    else if (pixelNewValue > 255) pixelNewValue = 255;
 
    // Set value of pixel
    P2D(outImg, pitch, row, col, int) = pixelNewValue;
} 
*/

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
    //    cout<<e1<<endl;
        //if(!e1) break;
        Pin[m]=e1;
        m++;
    }
}

void ReadPPM(int *R,int *G , int *B, char *name)
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
        R[m]=e1;
        file>>e1;
        G[m]=e1;
        file>>e1;
        B[m]=e1;
        m++;
    }
}
int* ReadSizeImg(char * name)
{
    int * dim= new int[2];
    int fil, col;
    string line,s1;
    ifstream file(name);
    getline(file,line);
    getline(file,line);
    file>>fil>>col;
    dim[0]=fil; dim[1]=col;   
    return dim;
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

void WritePPM(int * R, int* G,int *B, int fil , int cols, char *name)
{
    ofstream file(name);
    file<<"P3"<<endl;
    file<<"# Function ConvertRGBtoGray @eddyrene"<<endl;
    file<<fil<<" "<<cols<<endl;
    file<<255<<endl;
    int n = fil*cols;
    int i=0;
    while(i<n)
    {
        file<<R[i]<<endl;
        file<<G[i]<<endl;
        file<<B[i]<<endl;
        i++;
    }
}
void print_vect(int *V, int n){
    int i;
    for (i = 0; i < n; i++)
		printf("%d ", V[i]);
}
int main(int argc, char *argv[])
{
	int * R;//,*G,*B;
	int * sR;//,*sG,*sB;
    int * d_R;//,*d_G,*d_B;
    int * sd_R;//,*sd_G,*sd_B;
	int * order = ReadSizeImg("img.pgm");
	int N=order[0]; int M=order[1];

	int THREADS_PER_BLOCK = 16;
	int size =3*N*M*sizeof(int);

    cout<<"tamano Imagen "<<N<<" "<<M<<"  size "<<size<<endl;

    int k[9]={-1,0,1,-2,0,2,-1,0,1};
	int *d_k;

    cudaMalloc((void **)&d_R, size);
	//cudaMalloc((void **)&d_G, size);
	//cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&sd_R, size);
	//cudaMalloc((void **)&sd_G, size);
	//cudaMalloc((void **)&sd_B, size);
	cudaMalloc((void **)&d_k,9*sizeof(int));

   
    R = (int *)malloc(size);
   // G = (int *)malloc(size); 
    //B = (int *)malloc(size);
    ReadPPM(R,"img.pgm");
     cout<<"pasa"<<endl;
       // ;printf("\n Impriendo R \n");
    	//print_vect(R,order[0]*order[1]); printf("\nImpriendo B \n");
    	//print_vect(G,order[0]*order[1]);printf("\nImpriendo G \n");
    	//print_vect(B,order[0]*order[1]);
    sR = (int *)malloc(size); 
   // sG = (int *)malloc(size); 
    //sB = (int *)malloc(size); 

	//for(int i=0;i<N*N;i++)
	//	sR[i]=0;
	//print_vect(sR,order[0]*order[1]);
	cudaMemcpy(d_R, R, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_G, G, size, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_k, k, 9*sizeof(int), cudaMemcpyHostToDevice);




	int blocks= (N + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK;
	dim3 dimGrid(blocks, blocks, 1);
	dim3 dimBlock(THREADS_PER_BLOCK,THREADS_PER_BLOCK, 1);
	
	cout<<"blocks : \n"<<blocks<<"\n threds: \n "<<THREADS_PER_BLOCK<<endl; 
	convolution<<<dimGrid,dimBlock>>>(d_R, d_k ,sd_R,1, N, M);
    //convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,imageChannels, imageWidth, imageHeight);

      // convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
                                   //   imageChannels, imageWidth, imageHeight);
		//blurKernel<<<dimGrid,dimBlock>>>( d_Pout, d_Pin, N, M);
	cudaMemcpy(sR, sd_R, size, cudaMemcpyDeviceToHost);
	//cudaMemcpy(sG, sd_G, size, cudaMemcpyDeviceToHost);
	//cudaMemcpy(sB, sd_B, size, cudaMemcpyDeviceToHost);

	//printf("\n Impriendo R \n");
	//print_vect(sR,order[0]*order[1]); printf("\nImpriendo B \n");
	//print_vect(sG,order[0]*order[1]);printf("\nImpriendo G \n");
	//print_vect(sB,order[0]*order[1]);
	//WritePPM(sR,sG,sB,N,M,"convLena.ppm");  
    cout<<"ss"<<endl;
	WritePPM(sR, N,M,"siete.ppm");  
	free(R); //free(G);free(B);
	cudaFree(d_R); //cudaFree(d_B);cudaFree(d_G);
	cudaFree(sd_R); //cudaFree(sd_B);cudaFree(sd_G);
	return 0;
}
