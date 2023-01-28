#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp> 
#include <opencv2/calib3d/calib3d.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <sys/stat.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <time.h>
 
// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{7,11}; 
 
int main()
{
  // Creating vector to store vectors of 3D points for each checkerboard image
  std::vector<std::vector<cv::Point3f> > objpoints;
 
  // Creating vector to store vectors of 2D points for each checkerboard image
  std::vector<std::vector<cv::Point2f> > imgpoints;
 
  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for(int i{0}; i<CHECKERBOARD[1]; i++)
  {
    for(int j{0}; j<CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(j,i,0));
  }
 
 
  // Extracting path of individual image stored in a given directory
  std::vector<cv::String> images;
  // Path of the folder containing checkerboard images
  std::string path = "/home/tj/project/dissertation/image_set/frames_9/*.png";
 
  cv::glob(path, images);
 
  cv::Mat frame, gray;
  // vector to store the pixel coordinates of detected checker board corners 
  std::vector<cv::Point2f> corner_pts;
  bool success;
 
  // Looping over all the images in the directory
  for(int i {0}; i<images.size(); i++)
  {
    frame = cv::imread(images[i]);
    cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
 
    // Finding checker board corners
    // If desired number of corners are found in the image then success = true  
    success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
    
    /* 
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display 
     * them on the images of checker board
    */
    if(success)
    {
      cv::TermCriteria criteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.001);
       
      // refining pixel coordinates for given 2d points.
      cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
       
      // Displaying the detected corner points on the checker board
      cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
       
      objpoints.push_back(objp);
      imgpoints.push_back(corner_pts);
    }
 
    cv::imshow("Image",frame);
    cv::waitKey(0);
  }
 
  cv::destroyAllWindows();
 
  cv::Mat cameraMatrix = (cv::Mat1d(3, 3) << 623.0457, 0, 321.239, 0, 623.0457, 242.932, 0, 0, 1), distCoeffs,R,T;
  cv::TermCriteria(cv::CALIB_USE_INTRINSIC_GUESS | cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  /*
   * Performing camera calibration by 
   * passing the value of known 3D points (objpoints)
   * and corresponding pixel coordinates of the 
   * detected corners (imgpoints)
  */
  double rpr_err = cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);
 
  std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
  std::cout << "distCoeffs : " << distCoeffs << std::endl;
  std::cout << "Rotation vector : " << R << std::endl;
  std::cout << "Translation vector : " << T << std::endl;
  std::cout << "Reprojection error: " << rpr_err << std::endl;

  cv::Mat dst, map1, map2,new_camera_matrix;
  cv::Size imageSize(cv::Size(gray.cols,gray.rows));
  
  // Refining the camera matrix using parameters obtained by calibration
  new_camera_matrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0);
  
  // Method 1 to undistort the image
  cv::undistort( frame, dst, new_camera_matrix, distCoeffs, new_camera_matrix );
  
  // Method 2 to undistort the image
  cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),imageSize, CV_16SC2, map1, map2);
  
  cv::remap(frame, dst, map1, map2, cv::INTER_LINEAR);
  
  //Displaying the undistorted image
  cv::imshow("undistorted image",dst);
  cv::waitKey(0);;
 
  return 0;
}
