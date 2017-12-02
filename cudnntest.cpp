
#include <iostream>

#include <cuda.h>
#include <cudnn.h>
#include "error_util.h"

using namespace std;

int main() {
  const int minibatch_size = 1;
  const int feature_num = 2;
  const int in_size = 3;
  //float allsize = (float)minibatch_size*feature_num*in_size*in_size;
  float allsize = 6;

  // 入力
  float srcData[minibatch_size][feature_num][in_size][in_size] = {
    {
      { { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 } },
      { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } }
    }
  };

  float *srcData_dev;
  checkCudaErrors(cudaMalloc((void**)&srcData_dev, sizeof(srcData)));

  // Copy input vectors from host memory to GPU buffers.
  checkCudaErrors(cudaMemcpy(srcData_dev, srcData, sizeof(srcData), cudaMemcpyHostToDevice));


  // 出力
  float dstData[minibatch_size][feature_num][in_size][in_size] = {
    {
      { { 3, 1, 1 }, { 0, 1, 0 }, { 0, 1, 0 } },
      { { 10, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } }
    }
  };
  float *dstData_dev;
  checkCudaErrors(cudaMalloc((void**)&dstData_dev, sizeof(dstData)));
  checkCudaErrors(cudaMemcpy(dstData_dev, dstData, sizeof(dstData), cudaMemcpyHostToDevice));

  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;

  checkCUDNN(cudnnCreate(&cudnnHandle));
  checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));

  checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, minibatch_size, feature_num, in_size, in_size));
  checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, minibatch_size, feature_num, in_size, in_size));


  float alpha = 3.0f;
  float beta = 2.0f;
  checkCUDNN(cudnnAddTensor(cudnnHandle,
				     &alpha,
				     srcTensorDesc,
				     srcData_dev,
				     &beta,
				     dstTensorDesc,
				     dstData_dev));

  // Copy output vector from GPU buffer to host memory.
  checkCudaErrors(cudaMemcpy(dstData, dstData_dev, sizeof(dstData), cudaMemcpyDeviceToHost));

  for (int i = 0; i < feature_num; i++) {
    for (int y = 0; y < in_size; y++) {
      cout << "{";
      for (int x = 0; x < in_size; x++) {
	cout << dstData[0][i][y][x] << ", ";
      }
      cout << "}, ";
    }
    cout << endl;
  }
  checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
  checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
  checkCUDNN(cudnnDestroy(cudnnHandle));

  return 0;
}
;
