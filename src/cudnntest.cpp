#include <iostream>

#include <cuda.h>
#include <cudnn.h>
#include <array>
#include "error_util.h"
#include "cudnn_test.h"

   
void CudnnRun::cudnnAddTensor_run(float *in_data,float *out_data, int mb_size, int feature_num, int in_size) {

  float *in_data_dev;
  checkCudaErrors(cudaMalloc((void**)&in_data_dev, sizeof(in_data)));

  // Copy input vectors from host memory to GPU buffers.
  checkCudaErrors(cudaMemcpy(in_data_dev, in_data, sizeof(in_data), cudaMemcpyHostToDevice));


  // 出力
  float *out_data_dev;
  checkCudaErrors(cudaMalloc((void**)&out_data_dev, sizeof(out_data)));
  checkCudaErrors(cudaMemcpy(out_data_dev, out_data, sizeof(out_data), cudaMemcpyHostToDevice));

  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;

  checkCUDNN(cudnnCreate(&cudnnHandle));
  checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));

  checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, mb_size, feature_num, in_size, in_size));
  checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, mb_size, feature_num, in_size, in_size));


  float alpha = 3.0f;
  float beta = 2.0f;
  checkCUDNN(cudnnAddTensor(cudnnHandle,
			    &alpha,
			    srcTensorDesc,
			    in_data_dev,
			    &beta,
			    dstTensorDesc,
			    out_data_dev));

  // Copy output vector from GPU buffer to host memory.
  checkCudaErrors(cudaMemcpy(out_data, out_data_dev, sizeof(out_data), cudaMemcpyDeviceToHost));

  checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
  checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
  checkCUDNN(cudnnDestroy(cudnnHandle));
  
};
