//
// Created by caps on 1/10/26.
//

#ifndef ORB_SLAM2_LOGGER_H
#define ORB_SLAM2_LOGGER_H


#include <iostream>
#include <string>
#include <fstream>




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

template <typename T>
class Logger
{
public:
    static std::mutex logMutex;

    static void LogInfoThread(const T& msg)
    {
        //standard color
        std::string color = BRIGHT_WHITE_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoMapper(const T& msg)
    {
        //standard color
        std::string color = MAGENTA_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoWhite(const T& msg)
    {
        std::string color = WHITE_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoGray(const T& msg)
    {
        std::string color = GRAY_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoI(const T& msg)
    {
        //standard color
        std::string color = BLUE_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoII(const T& msg)
    {
        //standard color
        std::string color = WHITE_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoIII(const T& msg)
    {
        //standard color
        std::string color = GREEN_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogInfoIV(const T& msg)
    {
        //standard color
        std::string color = CYAN_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogTime(const T& msg)
    {
        //standard color
        std::string color = CYAN_TEXT;
        std::string prefix = color + "[TIME] ";
        log(prefix, msg);
    }

    static void LogWarning(const T& msg)
    {
        //standard color
        std::string color = YELLOW_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

    static void LogError(const T& msg)
    {
        //standard color
        std::string color = RED_TEXT;
        std::string prefix = color;
        log(prefix, msg);
    }

private:
    static void log(const std::string& prefix, const T& msg)
    {
        //auto now = std::chrono::system_clock::now();
        //std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::lock_guard<std::mutex> lock(logMutex);
        std::clog << prefix << msg << WHITE_TEXT << std::endl;
    }
};

template <typename T>
std::mutex Logger<T>::logMutex;
#endif //ORB_SLAM2_LOGGER_H