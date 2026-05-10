#pragma once
#include <opencv2/core/types.hpp>
namespace dart_vision {
struct GreenLight {
    float score = 0.;
    cv::Rect2f bbox; // bounding box in pixel coordinates
    cv::Point2f center; // center in pixel coordinates
    float radius;


    
};
} // namespace dart_vision