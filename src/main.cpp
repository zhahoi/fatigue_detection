#include <fatigue_detect.h>

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s <camera_index | video_path>\n", argv[0]);
        return -1;
    }

    // 分配人脸检测缓存
    unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if(!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        return -1;
    }

    // 加载 Dlib 68 点预测模型
    dlib::shape_predictor pose_model;
    try {
        dlib::deserialize("../model/shape_predictor_68_face_landmarks.dat") >> pose_model;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load shape predictor model: " << e.what() << std::endl;
        free(pBuffer);
        return -1;
    }

    // 打开摄像头或视频文件
    cv::VideoCapture cap;
    std::string src = argv[1];
    if (src.size() == 1 && std::isdigit(src[0])) {
        // 单字符数字视为摄像头索引
        int cam_idx = src[0] - '0';
        cap.open(cam_idx);
        std::cout << "Opening camera index " << cam_idx << "...\n";
    } else {
        // 否则当作视频文件路径
        cap.open(src);
        std::cout << "Opening video file " << src << "...\n";
    }

    if (!cap.isOpened()) {
        std::cerr << "Cannot open capture source: " << src << std::endl;
        free(pBuffer);
        return -1;
    }

    FatigueDetect detector;
    cv::Mat frame;
    while (true)
    {
        if (!cap.read(frame) || frame.empty())
        {
            // 视频读完 or 摄像头断流
            std::cout << "End of stream or cannot grab frame." << std::endl;
            break;
        }

        // 传入整帧进行疲劳检测并显示结果
        detector.detectFatigue(frame, pose_model, pBuffer);

        // 按 'q' 键退出
        char c = (char)cv::waitKey(1);
        if (c == 'q' || c == 'Q')
            break;
    }

    free(pBuffer);
    return 0;
}
