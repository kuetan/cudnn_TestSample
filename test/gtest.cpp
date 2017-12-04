#include "gtest/gtest.h" 
#include "cudnn_test.h"

class CudnnTest : public ::testing::Test {
protected:

  virtual void SetUp(){
  }

  virtual void TearDown(){
  }
};


TEST_F(CudnnTest, addtensor)
{
 const int mb_size = 1;
 const int feature_num = 2;
 const int in_size = 3;

 float in_data[mb_size][feature_num][in_size][in_size] = {
    {
      { { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 } },
      { { 2, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } }
    }
  };
 float out_data[mb_size][feature_num][in_size][in_size] = {
   {
     { { 3, 1, 1 }, { 0, 1, 0 }, { 0, 1, 0 } },
     { { 10, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } }
   }
 };

 float expect[3] = {6,5,1};
 
 CudnnRun m1 ;
 m1.cudnnAddTensor_run(&in_data[0][0][0][0],&out_data[0][0][0][0],mb_size,feature_num,in_size);
 
 for (int l = 0;l<in_size;l++) { 
   EXPECT_FLOAT_EQ(out_data[0][0][0][l], expect[l]);
 }
}

