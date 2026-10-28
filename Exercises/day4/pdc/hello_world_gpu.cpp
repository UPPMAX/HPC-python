/*
==============================================================================
Hello world
Copyright (C) 2023  Henric Zazzi <hzazzi@kth.se>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
==============================================================================
*/
#include <hip/hip_runtime.h>
#include <stdio.h>

// Kernel function that copies data string 1 -> string 2
__global__ void MyKernel(char* s1, char* s2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  s2[idx] = s1[idx];
  }

int main(int argc, char** argv) {
  char hosts1[] = "hello world";
  char hosts2[11], *devs1, *devs2;
  size_t size = sizeof(hosts1);
  int ndevices, ndevice;

  // Get number of GPUs available
  if (hipGetDeviceCount(&ndevices) != hipSuccess) {
    printf("No such devices\n");
    return 1;
    } 
  printf("You can access GPU devices: 0-%d\n", (ndevices - 1));
  ndevice = 0;
  if (argc > 1)
    ndevice = atoi(argv[1]);
  // Set default device to be used for subsequent hip API calls from this thread
  if (hipSetDevice(ndevice) != hipSuccess) {
    printf("Error initializing device %d\n", ndevice);
    return 1;
    }
  // Allocate memory on device
  hipMalloc(&devs1, size);
  hipMalloc(&devs2, size);
  // Copy data host -> device
  hipMemcpy(devs1, hosts1, size, hipMemcpyHostToDevice);
  // 3D-grid dimensions specifying the number of blocks to launch
  dim3 ngrid(1);
  // 3D-block dimensions specifying the number of threads in each block
  dim3 nblock(size);
  // Run kernel
  hipLaunchKernelGGL(MyKernel, ngrid, nblock, 0, 0, devs1, devs2);
  // Copy data device -> host
  hipMemcpy(hosts2, devs2, size, hipMemcpyDeviceToHost);
  // Free up memory
  hipFree(devs1);
  hipFree(devs2);
  printf("GPU %d: %s\n", ndevice, hosts2);
  }

