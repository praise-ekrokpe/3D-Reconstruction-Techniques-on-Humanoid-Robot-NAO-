#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;


int main()
{
    Mat img1, img2;
    img1 = imread("image-left.png", IMREAD_GRAYSCALE);
    img2 = imread("image-right.png", IMREAD_GRAYSCALE);

    double cm1[3][3] = {{621.4930113180801, 0, 321.2484043688772}, {0, 624.59821865034, 243.93182335470593}, {0, 0, 1}};
    double cm2[3][3] = {{621.4930113180801, 0, 321.2484043688772}, {0, 624.59821865034, 243.93182335470593}, {0, 0, 1}};
    double d1[1][5] = {{0.1305742880042869, -0.06283969925941815, 0.005934595032054272, -0.002533814218408031, -0.9717665994799077}};
    double d2[1][5] = {{0.1305742880042869, -0.06283969925941815, 0.005934595032054272, -0.002533814218408031, -0.9717665994799077}};

    Mat CM1 (3,3, CV_64FC1, cm1);
    Mat CM2 (3,3, CV_64FC1, cm2);
    Mat D1(1,5, CV_64FC1, d1);
    Mat D2(1,5, CV_64FC1, d2);

    double r[3][3] = {{1, 0, 0},{0, 1, 0 },{0, 0, 1}};
    double t[3][1] = {{0}, {0}, {0.2}};

    Mat R (3,3, CV_64FC1, r);
    Mat T (3,1, CV_64FC1, t);

    //Mat   R, T;
    Mat R1, R2, T1, T2, Q, P1, P2;

    stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY);

    Mat map11, map12, map21, map22;
    Size img_size = img1.size();
    initUndistortRectifyMap(CM1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(CM2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
    Mat img1r, img2r;
    remap(img1, img1r, map11, map12, INTER_LINEAR);
    remap(img2, img2r, map21, map22, INTER_LINEAR);

    cout << "P1:\n" << P1 << endl;
    cout << "P2:\n" << P2 << endl;
    cout << "Q:\n" << Q << endl;

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
    stereo->compute(img1,img2,disp);
  
    cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8UC1);  

    Mat points, points1;
    reprojectImageTo3D(disp, points, Q, false);
    cvtColor(points, points1, cv::COLOR_BGR2GRAY);

    ofstream point_cloud_file;
    point_cloud_file.open ("point_cloud.xyz");
    for(int i = 0; i < points.rows; i++) {
        for(int j = 0; j < points.cols; j++) {
            if(points.at<Vec3f>(i,j)[2] < 10) {
                point_cloud_file << points.at<Vec3f>(i,j)[0] << " " << points.at<Vec3f>(i,j)[1] << " " << points.at<Vec3f>(i,j)[2] 
                    << " " << static_cast<unsigned>(img1.at<uchar>(i,j)) << " " << static_cast<unsigned>(img1.at<uchar>(i,j)) << " " << static_cast<unsigned>(img1.at<uchar>(i,j)) << endl;
            }
        }
    }
    point_cloud_file.close();

    imshow("Img1", img1);
    imshow("Img2", img2);
    imshow("points", points);
    imshow("points1", points1);

    waitKey(0);

    return 0;
}