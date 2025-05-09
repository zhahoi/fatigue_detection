#ifndef _FATIGUE_DETECT_H_
#define _FATIGUE_DETECT_H_

#include <cstdint>
#include <fstream>
#include <string>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <chrono>
#include "facedetection/facedetectcnn.h"

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <typeinfo>

//define the buffer size. Do not change the size!
//0x9000 = 1024 * (16 * 2 + 4), detect 1024 face at most
#define DETECT_BUFFER_SIZE 0x9000

class FatigueDetect
{
public:
    void detectFatigue(cv::Mat& image, dlib::shape_predictor& pose_model, unsigned char* pBuffer);

private:
    double computeEAR(const std::vector<cv::Point>& eye);

    double computeMAR(const std::vector<cv::Point>& mouth);

    void extractFacialRegions(const dlib::full_object_detection& shape, int x_offset, int y_offset,
                        std::vector<cv::Point>& leftEye, std::vector<cv::Point>& rightEye, std::vector<cv::Point>& mouth);

    cv::Vec3d getHeadPose(const dlib::full_object_detection& shape, 
                        const cv::Mat& cam_matrix, 
                        const cv::Mat& dist_coeffs, 
                        const std::vector<cv::Point3f>& object_pts);

    void drawLandmarkPoints(cv::Mat& image, const std::vector<cv::Point>& points, 
                        const cv::Scalar& color = cv::Scalar(0, 255, 0), int radius = 2, int thickness = -1);

private:
    // 连续帧数判定
    const int CONSEC_FRAMES = 3;

    // EAR阈值和时间阈值（单位：秒）
    const double EYE_AR_THRESH = 0.20;
    const double EYE_CLOSED_TIME_THRESH = 2.0;
    // 状态记录变量
    bool eyeClosed = false;
    std::chrono::steady_clock::time_point eyeCloseStart;

    // MAR阈值和时间阈值（单位：秒）
    const double MOUTH_AR_THRESH = 0.65;
    const double MOUTH_OPEN_TIME_THRESH = 2.0;
    // 状态记录变量
    bool mouthOpen = false;
    std::chrono::steady_clock::time_point mouthOpenStart;

    // 点头持续时间阈值（秒）
    const double PITCH_NOD_THRESH = 25.0;  // 点头阈值（单位：角度）
    const double NOD_TIME_THRESH = 2.0;    // 点头持续时间阈值（秒）
    bool isNodding = false;
    std::chrono::steady_clock::time_point nodStart;

    // 头部姿态估计所需的3D参考点（世界坐标系）
    std::vector<cv::Point3f> object_pts = {
        cv::Point3f(6.825897, 6.760612, 4.402142),
        cv::Point3f(1.330353, 7.122144, 6.903745),
        cv::Point3f(-1.330353, 7.122144, 6.903745),
        cv::Point3f(-6.825897, 6.760612, 4.402142),
        cv::Point3f(5.311432, 5.485328, 3.987654),
        cv::Point3f(1.789930, 5.393625, 4.413414),
        cv::Point3f(-1.789930, 5.393625, 4.413414),
        cv::Point3f(-5.311432, 5.485328, 3.987654),
        cv::Point3f(2.005628, 1.409845, 6.165652),
        cv::Point3f(-2.005628, 1.409845, 6.165652),
        cv::Point3f(2.774015, -2.080775, 5.048531),
        cv::Point3f(-2.774015, -2.080775, 5.048531),
        cv::Point3f(0.000000, -3.116408, 6.097667),
        cv::Point3f(0.000000, -7.415691, 4.070434)
    };

    //Intrisics can be calculated using opencv sample code under opencv/sources/samples/cpp/tutorial_code/calib3d
    //Normally, you can also apprximate fx and fy by image width, cx by half image width, cy by half image height instead
    double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
    double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };

    cv::Mat cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
    cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);

    // 用于绘制3D立方体
    const std::vector<cv::Point3f> reprojectsrc = {
        { 10.0f,  10.0f,  10.0f},
        { 10.0f,  10.0f, -10.0f},
        { 10.0f, -10.0f, -10.0f},
        { 10.0f, -10.0f,  10.0f},
        {-10.0f,  10.0f,  10.0f},
        {-10.0f,  10.0f, -10.0f},
        {-10.0f, -10.0f, -10.0f},
        {-10.0f, -10.0f,  10.0f}
    };

    // 12 条边，每条由两个顶点索引组成
    const int line_pairs[12][2] = {
        {0,1},{1,2},{2,3},{3,0},
        {4,5},{5,6},{6,7},{7,4},
        {0,4},{1,5},{2,6},{3,7}
    };

    std::vector<cv::Point2f> reprojectdst;

    // 统计次数
    int eyeCloseCount = 0;
    int yawnCount = 0;
    int nodCount = 0;

    // 连续帧计数，用于“3帧触发一次事件”
    int eyeFrameCount   = 0;
    int mouthFrameCount = 0;
    int nodFrameCount   = 0;

    // 滑动窗口开始时间，用于一分钟内事件累积 
    std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
};

#endif