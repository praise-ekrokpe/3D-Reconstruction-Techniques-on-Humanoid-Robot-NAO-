#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv)
{

  cv::Mat imgL,imgR;
  
  imgL = cv::imread("image-left.png",0);
  cv::resize(imgL,imgL,cv::Size(600,600));
  imgR = cv::imread("image-right.png",0);
  cv::resize(imgR,imgR,cv::Size(600,600));
  
  int minDisparity = 0;
  int numDisparities = 64;
  int blockSize = 8;
  int disp12MaxDiff = 1;
  int uniquenessRatio = 10;
  int speckleWindowSize = 10;
  int speckleRange = 8;
  int mode = cv::StereoSGBM::MODE_SGBM;
  
  cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(minDisparity,numDisparities,blockSize,disp12MaxDiff,uniquenessRatio,speckleWindowSize,speckleRange,mode);

  cv::Mat disp;
  stereo->compute(imgL,imgR,disp);
  
  cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8U);
  disp.convertTo(disp, CV_32F, 1.0);
  disp = (disp/16.0f -(float)minDisparity)/((float)numDisparities);

  cv::imshow("Left image",imgL);
  cv::imshow("Right image",imgR);
  cv::imshow("disparity",disp);
  cv::waitKey(0);

  return 0;
}