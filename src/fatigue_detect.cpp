#include <fatigue_detect.h>

double FatigueDetect::computeEAR(const std::vector<cv::Point>& eye)
{
    if (eye.size() != 6) return 0.0;
    double A = norm(eye[1] - eye[5]);
    double B = norm(eye[2] - eye[4]);
    double C = norm(eye[0] - eye[3]);
    return (A + B) / (2.0 * C);
}

double FatigueDetect::computeMAR(const std::vector<cv::Point>& mouth)
{
    if (mouth.size() < 11) return 0.0;
    double A = norm(mouth[2] - mouth[10]);  // 51 - 59
    double B = norm(mouth[4] - mouth[8]);   // 53 - 57
    double C = norm(mouth[0] - mouth[6]);   // 49 - 55
    return (A + B) / (2.0 * C);
}

void FatigueDetect::extractFacialRegions(const dlib::full_object_detection& shape, int x_offset, int y_offset,
    std::vector<cv::Point>& leftEye, std::vector<cv::Point>& rightEye, std::vector<cv::Point>& mouth)
{
    for (unsigned int j = 0; j < shape.num_parts(); ++j)
    {
        int px = shape.part(j).x() + x_offset;
        int py = shape.part(j).y() + y_offset;
        cv::Point pt(px, py);

        // 获取左眼的关键点
        if (j >= 36 && j <= 41)
            leftEye.push_back(pt);
        // 获取右眼的关键点
        else if (j >= 42 && j <= 47)
            rightEye.push_back(pt);
        // 获取嘴巴的关键点
        else if (j >= 48 && j <= 67)
            mouth.push_back(pt);
    }
}

// 获取头部姿态（俯仰、偏航、滚转）
cv::Vec3d FatigueDetect::getHeadPose(const dlib::full_object_detection& shape, 
    const cv::Mat& cam_matrix, 
    const cv::Mat& dist_coeffs, 
    const std::vector<cv::Point3f>& object_pts)
{
    std::vector<cv::Point2d> image_pts;
    image_pts.push_back(cv::Point2d(shape.part(17).x(), shape.part(17).y())); // 左眉左上角
    image_pts.push_back(cv::Point2d(shape.part(21).x(), shape.part(21).y())); // 左眉右上角
    image_pts.push_back(cv::Point2d(shape.part(22).x(), shape.part(22).y())); // 右眉左上角
    image_pts.push_back(cv::Point2d(shape.part(26).x(), shape.part(26).y())); // 右眉右上角
    image_pts.push_back(cv::Point2d(shape.part(36).x(), shape.part(36).y())); // 左眼左上角
    image_pts.push_back(cv::Point2d(shape.part(39).x(), shape.part(39).y())); // 左眼右上角
    image_pts.push_back(cv::Point2d(shape.part(42).x(), shape.part(42).y())); // 右眼左上角
    image_pts.push_back(cv::Point2d(shape.part(45).x(), shape.part(45).y())); // 右眼右上角
    image_pts.push_back(cv::Point2d(shape.part(31).x(), shape.part(31).y())); // 鼻子左上角
    image_pts.push_back(cv::Point2d(shape.part(35).x(), shape.part(35).y())); // 鼻子右上角
    image_pts.push_back(cv::Point2d(shape.part(48).x(), shape.part(48).y())); // 嘴巴左上角
    image_pts.push_back(cv::Point2d(shape.part(54).x(), shape.part(54).y())); // 嘴巴右上角
    image_pts.push_back(cv::Point2d(shape.part(57).x(), shape.part(57).y())); // 嘴巴中央下角
    image_pts.push_back(cv::Point2d(shape.part(8).x(), shape.part(8).y()));   // 下巴角

    cv::Mat rvec, tvec;
    cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rvec, tvec);

    cv::projectPoints(reprojectsrc, rvec, tvec, cam_matrix, dist_coeffs, reprojectdst);

    cv::Mat rotation_mat;
    cv::Rodrigues(rvec, rotation_mat);

    cv::Mat pose_mat;
    cv::hconcat(rotation_mat, tvec, pose_mat);

    cv::Mat out_intrinsics, out_rotation, out_translation, euler_angle;
    cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, 
                    cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);

    double pitch = euler_angle.at<double>(0); // x
    double yaw   = euler_angle.at<double>(1); // y
    double roll  = euler_angle.at<double>(2); // z

    return cv::Vec3d(pitch, yaw, roll);
}

void FatigueDetect::drawLandmarkPoints(cv::Mat& image, const std::vector<cv::Point>& points, 
    const cv::Scalar& color, int radius, int thickness)
{
    for (const auto& pt : points)
    {
        cv::circle(image, pt, radius, color, thickness);
    }
}

void FatigueDetect::detectFatigue(cv::Mat& image, dlib::shape_predictor& pose_model, unsigned char* pBuffer)
{
    int* pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);

    cv::Mat& result_image = image;

    cv::putText(result_image, "Detected Faces: " + std::to_string(*pResults),
            cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);

    for(int i = 0; i < (pResults ? *pResults : 0); i++)
    {
        short *p = ((short*)(pResults + 1)) + 16 * i;
        int confidence = p[0];
        int x = p[1];
        int y = p[2];
        int w = p[3];
        int h = p[4];

        if (confidence < 70) continue;

        cv::rectangle(result_image, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
        cv::Rect faceRect = cv::Rect(x, y, w, h) & cv::Rect(0, 0, image.cols, image.rows);
        cv::Mat faceROI = image(faceRect);

        dlib::cv_image<dlib::bgr_pixel> dlib_img(faceROI);
        dlib::rectangle dlib_rect(0, 0, faceROI.cols - 1, faceROI.rows - 1);
        dlib::full_object_detection shape = pose_model(dlib_img, dlib_rect);

        std::vector<cv::Point> leftEyePts, rightEyePts, mouthPts;
        extractFacialRegions(shape, x, y, leftEyePts, rightEyePts, mouthPts);

        // 可视化68个关键点
        // for (unsigned int j = 0; j < shape.num_parts(); ++j)
        // {
        //     int px = shape.part(j).x() + x;
        //     int py = shape.part(j).y() + y;
        //     cv::circle(result_image, cv::Point(px, py), 2, cv::Scalar(0, 0, 255), -1);
        // }

        // EAR 检测
        if (leftEyePts.size() == 6 && rightEyePts.size() == 6)
        {
            double leftEAR = computeEAR(leftEyePts);
            double rightEAR = computeEAR(rightEyePts);
            double avgEAR = (leftEAR + rightEAR) / 2.0;
            cv::putText(result_image, "EAR: " + std::to_string(avgEAR).substr(0,4), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

            drawLandmarkPoints(result_image, leftEyePts);
            drawLandmarkPoints(result_image, rightEyePts);

            if (avgEAR < EYE_AR_THRESH)
            {
                eyeFrameCount++;
            }else{
                eyeFrameCount = 0;
                eyeClosed = false;
            }

            // 连续三帧，视为一次“眨眼事件”
            if (eyeFrameCount >= CONSEC_FRAMES)
            {
                eyeCloseCount++;
                eyeFrameCount = 0;
                eyeClosed = false;
            }

            // 当单次闭眼时长也超过阈值时，直接报警
            if (avgEAR < EYE_AR_THRESH && !eyeClosed) 
            {
                eyeClosed = true;
                eyeCloseStart = std::chrono::steady_clock::now();
            } else if (eyeClosed) {
                double dur = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - eyeCloseStart).count();
                if (dur >= EYE_CLOSED_TIME_THRESH) {
                    cv::putText(result_image, "ALERT: Eyes Closed Too Long!",
                        cv::Point(10, 75), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
                    eyeClosed = false;
                }
            }

            cv::putText(result_image, "Blinks: " + std::to_string(eyeCloseCount), cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,255), 2);
        }

        // MAR 检测
        if (mouthPts.size() >= 11)
        {
            double mar = computeMAR(mouthPts);
            putText(result_image, "MAR: " + std::to_string(mar).substr(0,4), cv::Point(10, 140), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

            drawLandmarkPoints(result_image, mouthPts);

            if (mar > MOUTH_AR_THRESH)
            {
                mouthFrameCount++;
            } else {
                mouthFrameCount = 0;
                mouthOpen = false;
            }

            // 连续三帧，视为一次“哈欠事件”
            if (mouthFrameCount >= CONSEC_FRAMES)
            {
                yawnCount++;
                mouthFrameCount = 0;
                mouthOpen = false;
            }

            // 当单次张嘴持续时长超过阈值时，直接报警
            if (mar < MOUTH_AR_THRESH && !mouthOpen) 
            {
                mouthOpen = true;
                mouthOpenStart = std::chrono::steady_clock::now();
            } else if (mouthOpen) {
                double dur = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - mouthOpenStart).count();
                if (dur >= MOUTH_OPEN_TIME_THRESH) {
                    cv::putText(result_image, "ALERT: Yawning Detected!",
                        cv::Point(10, 155), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
                    mouthOpen = false;
                }
            }

            cv::putText(result_image, "Yawns: " + std::to_string(yawnCount), cv::Point(10, 170), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,255), 2);
        }

        // 头部姿态 Pitch（点头检测）
        cv::Vec3d eulerAngles = getHeadPose(shape, cam_matrix, dist_coeffs, object_pts);
        double pitch = eulerAngles[0]; // 俯仰角
        cv::putText(result_image, "Pitch: " + std::to_string(pitch).substr(0,5), cv::Point(10, 220), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

        // 绘制立方体的12条边
        for (int i = 0; i < 12; ++i)
        {
            cv::Point2f p1 = reprojectdst[line_pairs[i][0]];
            cv::Point2f p2 = reprojectdst[line_pairs[i][1]];

            p1.x += x;  p1.y += y;
            p2.x += x;  p2.y += y;
            
            cv::line(
                result_image,
                p1, p2,
                cv::Scalar(0, 0, 255), 2, cv::LINE_AA
            );
        }

        if (pitch > PITCH_NOD_THRESH)
        {
            nodFrameCount++;
        } else {
            nodFrameCount = 0;
            isNodding = false;
        }

        // 连续三帧，视为一次“点头事件”
        if (nodFrameCount >= CONSEC_FRAMES)
        {
            nodCount++;
            nodFrameCount = 0;
            isNodding = false;
        }

        // 单词点头持续时长检测
        if (pitch > PITCH_NOD_THRESH && !isNodding) {
            isNodding = true;
            nodStart = std::chrono::steady_clock::now();
        } else if (isNodding) {
            double dur = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - nodStart).count();
            if (dur >= NOD_TIME_THRESH) {
                cv::putText(result_image, "ALERT: Nodding Detected!",
                    cv::Point(10, 235), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
                isNodding = false;
            }
        }

        cv::putText(result_image, "Nods: " + std::to_string(nodCount), cv::Point(10, 250), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,255), 2);
    }

    // 每60s判断一次疲劳
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - windowStart).count();
    if (elapsed >= 60.0)
    {
        bool f_eye = eyeCloseCount > 25;
        bool f_yawn = yawnCount > 5;
        bool f_nod = nodCount > 6;
        if (f_eye || f_yawn || f_nod)
        {
            cv::putText(result_image, ">>> DRIVER FATIGUE <<<", cv::Point(10, 310), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,255), 3);
        }

        // 重置计数和窗口
        eyeCloseCount = yawnCount = nodCount = 0;
        windowStart = now;
    }

    cv::imshow("Fatigue Detection with Landmarks", result_image);
}
