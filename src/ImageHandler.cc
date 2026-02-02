//
// Created by caps on 1/9/26.
//
#include "ImageHandler.h"



float ORB_SLAM2::ImageHandler::bilinearInterpolation(const cv::Mat& I, float u, float v)
{
    int x = (int)std::floor(u);
    int y = (int)std::floor(v);
    if (x<0||y<0||x+1>=I.cols||y+1>=I.rows) return 0.f;
    float du = u - x;
    float dv = v - y;
    const float* r0 = I.ptr<float>(y);
    const float* r1 = I.ptr<float>(y+1);
    float I00=r0[x];
    float I10=r0[x+1];
    float I01=r1[x];
    float I11=r1[x+1];
    return (1-du)*(1-dv)*I00 + du*(1-dv)*I10 + (1-du)*dv*I01 + du*dv*I11;
}

void ORB_SLAM2::ImageHandler::bilinearInterpolationGrad(const cv::Mat& Ix, const cv::Mat& Iy, float u, float v,
    float& gx, float& gy)
{
    int x = (int)std::floor(u);
    int y = (int)std::floor(v);
    if (x<0||y<0||x+1>=Ix.cols||y+1>=Ix.rows)
        { gx=gy=0.f; return; }
    float du = u - x;
    float dv = v - y;
    const float* x0 = Ix.ptr<float>(y);
    const float* x1 = Ix.ptr<float>(y+1);
    const float* y0 = Iy.ptr<float>(y);
    const float* y1 = Iy.ptr<float>(y+1);
    float gx00=x0[x];
    float gx10=x0[x+1];
    float gx01=x1[x];
    float gx11=x1[x+1];
    float gy00=y0[x];
    float gy10=y0[x+1];
    float gy01=y1[x];
    float gy11=y1[x+1];
    gx = (1-du)*(1-dv)*gx00 + du*(1-dv)*gx10 + (1-du)*dv*gx01 + du*dv*gx11;
    gy = (1-du)*(1-dv)*gy00 + du*(1-dv)*gy10 + (1-du)*dv*gy01 + du*dv*gy11;
}

cv::Matx44f ORB_SLAM2::ImageHandler::se3exp(const cv::Matx<float,6,1>& xi)
{
    cv::Vec3f w(xi(0), xi(1), xi(2));   // omega
    cv::Vec3f v(xi(3), xi(4), xi(5));   // v (translation twist)

    float th = cv::norm(w);
    cv::Matx33f I = cv::Matx33f::eye();
    cv::Matx33f W(   0,   -w[2],  w[1],
                   w[2],     0,  -w[0],
                  -w[1],  w[0],     0 );
    cv::Matx33f W2 = W * W;

    cv::Matx33f R = I, V = I;

    if (th > 1e-8f)
    {
        float s_over_th   = std::sin(th) / th;
        float one_mc_over = (1.f - std::cos(th)) / (th*th);
        float th_ms_over  = (th - std::sin(th)) / (th*th*th);

        R = I + s_over_th * W + one_mc_over * W2;
        V = I + one_mc_over * W + th_ms_over * W2;
    }
    else
    {
        // series: R ≈ I + W,  V ≈ I + 0.5 W + (1/6) W^2
        R = I + W;
        V = I + 0.5f * W + (1.f/6.f) * W2;
    }

    cv::Vec3f t = V * v;

    cv::Matx44f T = cv::Matx44f::eye();
    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
            T(i,j) = R(i,j);
    T(0,3) = t[0]; T(1,3) = t[1]; T(2,3) = t[2];
    return T;
}

