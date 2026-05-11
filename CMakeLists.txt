cmake_minimum_required(VERSION 3.5)
project(dart_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Boost 1.58 必须明确组件
find_package(Boost 1.58 REQUIRED COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(HikSDK REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/3rdparty/Hungarian
    ${PROJECT_SOURCE_DIR}/3rdparty/KalmanHyLib
)

set(SOURCES
    src/main.cpp
    3rdparty/Hungarian/Hungarian.cpp
)

# 编译库
add_library(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    pthread
    Eigen3::Eigen
    HikSDK::HikSDK
    yaml-cpp
    ceres
)

# 编译可执行文件
add_executable(vision src/main.cpp)
target_link_libraries(vision ${PROJECT_NAME})