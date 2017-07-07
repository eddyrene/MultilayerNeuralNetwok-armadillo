#include <fstream>
#include <iostream>
using namespace std;

#define Mask_width  3
#define Mask_radius Mask_width/2
#define TILE_WIDTH 32
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0), 255))

__global__ 
void convolution(double *I, const int* __restrict__ M, double *P, int channels, int width, int height) 
{
   __shared__ double N_ds[w][w];
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

void WritePGM(double * Pout, int fil , int cols,const  char *name)
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
        file<<(int)Pout[i]<<endl;
        i++;
    }
}
void kernel_convolution(double *R , double *sR , int order ,int chanel )
{
   double * d_R;//,*d_G,*d_B;
   double * sd_R;//,*sd_G,*sd_B;
   int N=order;
   int M=order;

   int THREADS_PER_BLOCK = 32;
   int size =1*N*M*sizeof(double);
    //cout<<"tamano Imagen "<<N<<" "<<M<<"  size "<<size<<endl;
   int k[9]={-1,0,1,-2,0,2,-1,0,1};
   int *d_k;

   cudaMalloc((void **)&d_R, size);
   cudaMalloc((void **)&sd_R, size);
   cudaMalloc((void **)&d_k,9*sizeof(int));   
   
   cudaMemcpy(d_R, R, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_k, k, 9*sizeof(int), cudaMemcpyHostToDevice);

   int blocks= (N + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK;
   dim3 dimGrid(blocks, blocks, 1);
   dim3 dimBlock(THREADS_PER_BLOCK,THREADS_PER_BLOCK, 1);
      convolution<<<dimGrid,dimBlock>>>(d_R,d_k,sd_R,chanel, N, M);
   cudaMemcpy(sR, sd_R, size, cudaMemcpyDeviceToHost);
   string name = "result.ppm";
   WritePGM(sR, N,M,name.c_str());  
   //free(R); //free(G);free(B);
   cudaFree(d_R); //cudaFree(d_B);cudaFree(d_G);
   cudaFree(sd_R); //cudaFree(sd_B);cudaFree(sd_G);
   //return 0;
}