//
// Created by caps on 1/9/26.
//

#ifndef ORB_SLAM2_IMAGEHANDLER_H
#define ORB_SLAM2_IMAGEHANDLER_H

#include <opencv2/core.hpp>

namespace ORB_SLAM2
{
    class ImageHandler
    {
    public:
        static float bilinearInterpolation(const cv::Mat& image, float u, float v);
        static void bilinearInterpolationGrad(const cv::Mat& Ix,const cv::Mat& Iy, float u, float v, float& gx, float& gy);
        static cv::Matx44f se3exp(const cv::Matx<float,6,1>& xi);
    };
}
#endif //ORB_SLAM2_IMAGEHANDLER_H