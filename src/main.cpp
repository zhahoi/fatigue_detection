#include <fatigue_detect.h>

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        printf("Usage: %s <camera index>\n", argv[0]);
        return -1;
    }

    unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if(!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        return -1;
    }

    dlib::shape_predictor pose_model;
    try {
        dlib::deserialize("../model/dlib_point.dat") >> pose_model;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load shape predictor model: " << e.what() << std::endl;
        return -1;
    }

    cv::VideoCapture cap;
    if(isdigit(argv[1][0])) {
        cap.open(argv[1][0]-'0');
    }

    if(!cap.isOpened()) {
        std::cerr << "Cannot open the camera." << std::endl;
        return -1;
    }

    FatigueDetect fatigueDetector;

    while(cap.isOpened())
    {
        cv::Mat image;
        cap >> image;
        if (image.empty()) continue;

        fatigueDetector.detectFatigue(image, pose_model, pBuffer);

        if((cv::waitKey(2) & 0xFF) == 'q') break;
    }

    free(pBuffer);
    return 0;
}
