//float cudnnAddTensor_test(float *in_data,int mb_size,int feature_num,int in_size);
class CudnnRun
{
public:
  //CudnnRun(float *in_data,float *out_data,int mb_size,int feature_num,int in_size);
  void cudnnAddTensor_run(
			  float *in_data ,
			  float *out_data,
			  int mb_size,
			  int feature_num,
			  int in_size);

};

