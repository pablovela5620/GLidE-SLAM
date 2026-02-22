//
// Created by caps on 2/21/26.
//

#ifndef GLIDE_SLAM_MOTIONMODEL_H
#define GLIDE_SLAM_MOTIONMODEL_H

#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
struct MotionModel
{
    void setParams(float alpha, float beta, float gamma, float maxT, float maxA)
    {
        m_alpha = alpha;
        m_beta = beta;
        m_gamma = gamma;
        m_maxT = maxT;
        m_maxA = maxA;
    }
    float m_alpha = 0.2f;
    float m_beta = 0.1f;
    float m_gamma = 0.01f;
    float m_maxT = 2.0f;
    float m_maxA = 90.0f;

    struct Twist
    {
        glm::vec3 t;  // linear velocity/acceleration
        glm::vec3 w;  // angular velocity/acceleration

        Twist() : t(0,0,0), w(0,0,0) {}
    };

    bool ready{false};

    // State
    glm::mat4 T;   // current pose
    Twist v;       // velocity
    Twist a;       // acceleration

    MotionModel() : T(glm::mat4(1.0f)) {}

    glm::mat3 skew(const glm::vec3& w)
    {
        glm::mat3 W(0.0f);
        W[0][1] = -w.z;
        W[0][2] =  w.y;
        W[1][0] =  w.z;
        W[1][2] = -w.x;
        W[2][0] = -w.y;
        W[2][1] =  w.x;
        return W;
    }

    glm::mat4 se3exp(const glm::vec3& w, const glm::vec3& v)
    {
        float th = glm::length(w);

        glm::mat3 I(1.0f);
        glm::mat3 W = skew(w);
        glm::mat3 W2 = W * W;

        glm::mat3 R = I;
        glm::mat3 V = I;

        if (th > 1e-8f)
        {
            float th2 = th * th;
            float th3 = th2 * th;

            float s_over_th   = std::sin(th) / th;
            float one_mc_over = (1.0f - std::cos(th)) / th2;
            float th_ms_over  = (th - std::sin(th)) / th3;

            R = I + s_over_th * W + one_mc_over * W2;
            V = I + one_mc_over * W + th_ms_over * W2;
        }
        else
        {
            R = I + W;
            V = I + 0.5f * W + (1.0f/6.0f) * W2;
        }

        glm::vec3 t_new = V * v;

        glm::mat4 T_out(1.0f);

        // GLM is column-major: T[col][row]
        T_out[0][0]=R[0][0]; T_out[0][1]=R[0][1]; T_out[0][2]=R[0][2];
        T_out[1][0]=R[1][0]; T_out[1][1]=R[1][1]; T_out[1][2]=R[1][2];
        T_out[2][0]=R[2][0]; T_out[2][1]=R[2][1]; T_out[2][2]=R[2][2];

        T_out[3][0]=t_new.x;
        T_out[3][1]=t_new.y;
        T_out[3][2]=t_new.z;

        return T_out;
    }

    glm::mat4 expSE3(const Twist& xi)
    {
        return se3exp(xi.w, xi.t);
    }

    Twist logSE3(const glm::mat4& T_in)
    {
        Twist xi;

        glm::mat3 R = glm::mat3(T_in);
        glm::vec3 t = glm::vec3(T_in[3]);

        float trace = R[0][0] + R[1][1] + R[2][2];
        float c = (trace - 1.0f) * 0.5f;
        c = glm::clamp(c, -1.0f, 1.0f);

        float theta = std::acos(c);

        if (theta < 1e-8f)
        {
            xi.w = glm::vec3(0,0,0);
            xi.t = t;
            return xi;
        }

        float sin_theta = std::sin(theta);

        glm::mat3 W = (theta / (2.0f * sin_theta)) * (R - glm::transpose(R));
        xi.w = glm::vec3(W[2][1], W[0][2], W[1][0]);

        glm::mat3 I(1.0f);
        glm::mat3 W2 = W * W;

        float theta2 = theta * theta;
        float A = (1.0f - std::cos(theta)) / theta2;
        float B = (theta - std::sin(theta)) / (theta2 * theta);

        glm::mat3 V_inv = I - 0.5f * W + (1.0f/theta2) *
            (1.0f - (theta * sin_theta) / (2.0f * (1.0f - std::cos(theta)))) * W2;

        xi.t = V_inv * t;

        return xi;
    }

    glm::mat4 predict(float dt)
    {
        Twist xi;
        xi.t = v.t * dt + 0.5f * a.t * dt * dt;
        xi.w = v.w * dt + 0.5f * a.w * dt * dt;

        return T * expSE3(xi);
    }

    Twist innovation(const glm::mat4& T_pred, const glm::mat4& T_meas)
    {
        glm::mat4 dT = glm::inverse(T_pred) * T_meas;
        return logSE3(dT);
    }

    bool gate(const Twist& delta, float dt, float zRef)
    {
        //we normalize by the scene depth
        float trans = glm::length(delta.t)/std::max(zRef,1e-6f);
        float rot   = glm::length(delta.w);

        float maxTrans = m_maxT * dt;
        float maxRot   = glm::radians(m_maxA) * dt;

        return (trans <= maxTrans) && (rot <= maxRot);
    }

    void update(const cv::Mat& pose, float dt, float zRef)
    {
        if (dt < 1e-6f)
            return;


        glm::mat4 m;
        for (size_t r = 0; r < 4; ++r)
            for (size_t c = 0; c < 4; ++c)
                m[c][r] = pose.at<float>((int)r, (int)c);

        if (!ready)
        {
            reset(m);
            ready = true;
            return;
        }



        glm::mat4 T_pred = predict(dt);
        Twist delta = innovation(T_pred, m);

        if (!gate(delta, dt,zRef))
        {
            T = T_pred;
            return;
        }

        // Pose correction
        Twist corr;
        corr.t = m_alpha * delta.t;
        corr.w = m_alpha * delta.w;
        T = T_pred * expSE3(corr);

        // Measured velocity
        Twist v_meas;
        v_meas.t = delta.t / dt;
        v_meas.w = delta.w / dt;

        // Measured acceleration
        Twist a_meas;
        a_meas.t = (v_meas.t - v.t) / dt;
        a_meas.w = (v_meas.w - v.w) / dt;

        // Update velocity
        v.t = (1.0f - m_beta)  * (v.t + a.t * dt) + m_beta  * v_meas.t;
        v.w = (1.0f - m_beta)  * (v.w + a.w * dt) + m_beta  * v_meas.w;

        // Update acceleration
        a.t = (1.0f - m_gamma) * a.t + m_gamma * a_meas.t;
        a.w = (1.0f - m_gamma) * a.w + m_gamma * a_meas.w;

        // Clamp acceleration to avoid explosions
        float maxAcc = 10.0f;
        if (glm::length(a.t) > maxAcc)
            a.t = glm::normalize(a.t) * maxAcc;

        float maxAngAcc = glm::radians(720.0f);
        if (glm::length(a.w) > maxAngAcc)
            a.w = glm::normalize(a.w) * maxAngAcc;
    }

    void checkResidual(float dt, const cv::Mat& pose, float zRef)
    {
        if (dt <= 1e-6f || pose.empty() || pose.type() != CV_32FC1)
            return;

        glm::mat4 T_meas;
        for (size_t r = 0; r < 4; ++r)
            for (size_t c = 0; c < 4; ++c)
                T_meas[c][r] = pose.at<float>((int)r, (int)c);

        glm::mat4 T_pred = predict(dt);
        Twist delta = innovation(T_pred, T_meas);

        // monocular: translation is in arbitrary scale units -> log both raw and normalized
        const float trans = glm::length(delta.t);
        const float transNorm = trans / std::max(zRef, 1e-6f);
        const float rotDeg = glm::degrees(glm::length(delta.w));

        const float gateT = m_maxT * dt;
        const float gateRDeg = glm::degrees(glm::radians(m_maxA) * dt);

        std::cout << "Motion residual: "
                  << "trans=" << trans
                  << " transNorm=" << transNorm
                  << " rotDeg=" << rotDeg
                  << " gateT=" << gateT
                  << " gateRDeg=" << gateRDeg
                  << std::endl;
    }

    void reset(const glm::mat4& T_init)
    {
        T = T_init;
        v = Twist();
        a = Twist();
    }
};

#endif //GLIDE_SLAM_MOTIONMODEL_H