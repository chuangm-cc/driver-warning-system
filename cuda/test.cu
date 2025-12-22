#include <iostream>
// #include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <chrono>


// Kernel function to add the elements of two arrays
__global__ void addNums(int *output, int *x, int *y, int num_iters) {
  for (int i = 0; i < num_iters; i++) {
    output[i] = x[i] + y[i];
  }
}

int main() {
  // Declare the variables
  int num_iters = 120;
  int *x;
  int *y;
  int *output;
  
  // Seeding the random number generator
  srand(10);

  std::cout << "Hello World, this is CUDA sample code" << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();

  // Declare the memory size for the variables from the unified memory location accessible from CPU or GPU
  cudaMallocManaged(&x, num_iters*sizeof(int));
  cudaMallocManaged(&y, num_iters*sizeof(int));
  cudaMallocManaged(&output, num_iters*sizeof(int));

  // Initialization with random numbers
  for (unsigned int i = 0; i < num_iters; i++) {
    output[i] = 0;
    x[i] = rand() % 100;
    y[i] = rand() % 100;
  }

  // Run the kernel function on each 64 threads on 32 blocks of the GPU
  addNums<<<4, 128>>>(output, x, y, num_iters);

  // Synchronization between the CPU and GPU (CPU waiting for GPU to finish before accessing the memory)
  cudaDeviceSynchronize();
  std::cout << output[num_iters-1] << " " << x[num_iters-1] << " " << y[num_iters-1] << std::endl;

  // Releasing the memory
  cudaFree(x);
  cudaFree(y);

  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;
  std::cout << "Code Execution Completed" << std::endl;  

  return 0;
}
