//
// Created by caps on 2/16/26.
//

#ifndef GLIDE_SLAM_GLIDEUTILS_H
#define GLIDE_SLAM_GLIDEUTILS_H
//
// Created by caps on 1/10/26.
//

#include <iostream>
#include <string>
#include <fstream>
#include <mutex>

//OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

//maths library
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "GLideSettings.h"


#define BLACK_TEXT   "\033[30m"
#define RED_TEXT     "\033[31m"
#define GREEN_TEXT   "\033[32m"
#define YELLOW_TEXT  "\033[33m"
#define BLUE_TEXT    "\033[34m"
#define MAGENTA_TEXT "\033[35m"
#define CYAN_TEXT    "\033[36m"
#define WHITE_TEXT   "\033[97m"
#define GRAY_TEXT   "\033[37m"
#define RESET_TEXT   "\033[0m"

#define BRIGHT_BLACK_TEXT   "\033[90m"
#define BRIGHT_RED_TEXT     "\033[91m"
#define BRIGHT_GREEN_TEXT   "\033[92m"
#define BRIGHT_YELLOW_TEXT  "\033[93m"
#define BRIGHT_BLUE_TEXT    "\033[94m"
#define BRIGHT_MAGENTA_TEXT "\033[95m"
#define BRIGHT_CYAN_TEXT    "\033[96m"
#define BRIGHT_WHITE_TEXT   "\033[97m"


#define WIN_FOREGROUND_WHITE (FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)
#define WIN_FOREGROUND_YELLOW (FOREGROUND_RED | FOREGROUND_GREEN)

class Logger
{
public:
    static std::mutex logMutex;

    static void LogInfoThread(const std::string& msg)
    {
        //standard color
        std::string color = BRIGHT_WHITE_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoMapper(const std::string& msg)
    {
        //standard color
        std::string color = MAGENTA_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoWhite(const std::string& msg)
    {
        std::string color = WHITE_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoGray(const std::string& msg)
    {
        std::string color = GRAY_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoI(const std::string& msg)
    {
        //standard color
        std::string color = BLUE_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoII(const std::string& msg)
    {
        //standard color
        std::string color = WHITE_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoIII(const std::string& msg)
    {
        //standard color
        std::string color = GREEN_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoIV(const std::string& msg)
    {
        //standard color
        std::string color = CYAN_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogTime(const std::string& msg)
    {
        //standard color
        std::string color = CYAN_TEXT;
        std::string prefix = color + "[TIME] ";
        log(prefix, msg);
    }

    static void LogWarning(const std::string& msg)
    {
        //standard color
        std::string color = YELLOW_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogError(const std::string& msg)
    {
        //standard color
        std::string color = RED_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

private:
    static void log(const std::string& color, const std::string& msg)
    {
        std::lock_guard<std::mutex> lock(logMutex);
        std::clog << color << msg << RESET_TEXT << '\n';
    }
};

namespace GLideUtils
{
    glm::vec3 readInVector(cv::FileStorage& fs, const std::string& parameter);
    glm::vec4 readInVector(cv::FileStorage& fs, const std::string& parameter, float defaultW);
    bool ReadConfigFile(const std::string& path, GLideSettings* GLideSettings);
}
#endif //GLIDE_SLAM_GLIDEUTILS_H