#pragma once

#include <sstream>
#include <stdio.h>

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure\nError: " << cudnnGetErrorString(status); \
      printf("[%d] CudaError:%s\n",__LINE__,cudnnGetErrorString(status)); \
      exit(status);\
    }                                                                  \
  }

#define checkCudaErrors(status) {                                      \
    if (status != 0) {                                                 \
      printf("[%d]CudaError:%s\n",__LINE__,cudaGetErrorString(status)); \
    }                                                                  \
  }
