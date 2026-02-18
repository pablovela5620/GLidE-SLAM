//
// Created by caps on 2/16/26.
//
#include "GLideUtils.h"

std::mutex Logger::logMutex;


glm::vec3 GLideUtils::readInVector(cv::FileStorage& fs, const std::string& parameter)
{
    std::vector<float> v;
    cv::FileNode fn = fs[parameter];

    if (fn.empty() || !fn.isSeq())
        std::cout << "Failed to read" + parameter + " from file." << std::endl;

    fn >> v;
    return glm::vec3(v[0], v[1], v[2]);
}
glm::vec4 GLideUtils::readInVector(cv::FileStorage& fs, const std::string& parameter, float defaultW)
{
    std::vector<float> v;
    cv::FileNode fn = fs[parameter];

    if (fn.empty() || !fn.isSeq())
        std::cout << "Failed to read" + parameter + " from file." << std::endl;

    fn >> v;

    float w = (v.size() >= 4) ? v[3] : defaultW;
    return glm::vec4(v[0], v[1], v[2], w);
}
bool GLideUtils::ReadConfigFile(const std::string &path, GLideSettings *GLideSettings)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        Logger::LogError("Failed to open configuration file.");
        return false;
    }

    //Direct tracking params
    GLideSettings->directTrackParams.patchSize      = fs["DirTrackParams.patchSize"];
    GLideSettings->directTrackParams.nLevels        = fs["DirTrackParams.nLevels"];
    GLideSettings->directTrackParams.scaleFactor    = fs["DirTrackParams.scaleFactor"];

    GLideSettings->directTrackParams.maxPoints      = fs["DirTrackParams.maxPoints"];
    GLideSettings->directTrackParams.enableAlign    = fs["DirTrackParams.enableAlign"];
    GLideSettings->directTrackParams.searchRadius   = fs["DirTrackParams.searchRadius"];
    GLideSettings->directTrackParams.huberK        = fs["DirTrackParams.huberK"];
    GLideSettings->directTrackParams.epsNorm        = fs["DirTrackParams.epsNorm"];
    GLideSettings->directTrackParams.minPoints        = fs["DirTrackParams.minPoints"];
    GLideSettings->directTrackParams.maxIterations  = fs["DirTrackParams.maxIterations"];

    // vec4 (per-level thresholds)
    GLideSettings->directTrackParams.searchThreshold = readInVector(fs, "DirTrackParams.searchThreshold", 0.0f);
    GLideSettings->directTrackParams.rejectThreshold = readInVector(fs, "DirTrackParams.rejectThreshold", 0.0f);
    GLideSettings->directTrackParams.maxShift        = readInVector(fs, "DirTrackParams.maxShift", 0.0f);


    //read in viewer slamSettings
    GLideSettings->gpuEngineParams.runViewer = fs["Viewer.displayWindow"];
    GLideSettings->gpuEngineParams.logTiming = fs["Viewer.logTiming"];
    GLideSettings->gpuEngineParams.width = fs["Viewer.width"];
    GLideSettings->gpuEngineParams.height = fs["Viewer.height"];
    GLideSettings->gpuEngineParams.windowFramesTitle = static_cast<std::string>(fs["Viewer.windowFramesTitle"]);
    GLideSettings->gpuEngineParams.windowMapTitle = static_cast<std::string>(fs["Viewer.windowMapTitle"]);
    GLideSettings->gpuEngineParams.scaleFactor = fs["Viewer.scaleFactor"];
    GLideSettings->gpuEngineParams.camMoveFactor = fs["Viewer.mouseMoveFactor"];
    GLideSettings->gpuEngineParams.cameraFollow = fs["Viewer.cameraFollow"];
    GLideSettings->gpuEngineParams.followDistance = fs["Viewer.followDistance"];

    GLideSettings->gpuEngineParams.currentKeyFrameColor = readInVector(fs, "Viewer.currentKeyFrameColor");
    GLideSettings->gpuEngineParams.allKeyFrameColor = readInVector(fs, "Viewer.allKeyFrameColor");
    GLideSettings->gpuEngineParams.tweenFrameDirectColor = readInVector(fs, "Viewer.tweenFrameDirectColor");
    GLideSettings->gpuEngineParams.tweenFrameColor = readInVector(fs, "Viewer.tweenFrameColor");
    GLideSettings->gpuEngineParams.mapPointsColor = readInVector(fs, "Viewer.mapPointsColor");
    GLideSettings->gpuEngineParams.mapPointsRefColor = readInVector(fs, "Viewer.mapPointsRefColor");
    GLideSettings->gpuEngineParams.featureLinesColor = readInVector(fs, "Viewer.featureLinesColor");

    GLideSettings->gpuEngineParams.fov = fs["Viewer.fov"];
    GLideSettings->gpuEngineParams.far = fs["Viewer.far"];
    GLideSettings->gpuEngineParams.near = fs["Viewer.near"];
    GLideSettings->gpuEngineParams.featuresMaxDepth = fs["Viewer.featuresMaxDepth"];

    return true;
}
