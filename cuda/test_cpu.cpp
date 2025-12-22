#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <chrono>

// Kernel function to add the elements of two arrays
void addNums(int *output, int *x, int *y, int num_iters) {
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
  x = new int[num_iters];
  y = new int[num_iters];
  output = new int[num_iters];

  // Initialization with random numbers
  for (unsigned int i = 0; i < num_iters; i++) {
    output[i] = 0;
    x[i] = rand() % 100;
    y[i] = rand() % 100;
  }

  addNums(output, x, y, num_iters);
  std::cout << output[num_iters-1] << " " << x[num_iters-1] << " " << y[num_iters-1] << std::endl;

  delete[] x;
  delete[] y;
  delete[] output;

  auto t2 = std::chrono::high_resolution_clock::now();

  std::cout << "execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;
  std::cout << "Code Execution Completed" << std::endl;  

  return 0;
}

