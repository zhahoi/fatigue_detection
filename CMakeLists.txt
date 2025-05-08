cmake_minimum_required (VERSION 3.10)

project (fatigue_detect)

set(OpenCV_DIR "/home/hit/Softwares/opencv-3.4.10/build")
set(dlib_DIR "/home/hit/Softwares/dlib/build")

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    include
    /home/hit/facial_landmark/facedetection/include
    ${dlib_INCLUDE_DIRS}  
)

add_executable(detect src/fatigue_detect.cpp src/main.cpp)
set_target_properties(detect PROPERTIES CXX_STANDARD 11)

target_link_libraries(detect
    ${OpenCV_LIBS}
    ${dlib_LIBRARIES}  
    /home/hit/pfld_ncnn/facedetection/lib/libfacedetection.so
)
