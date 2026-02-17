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
bool GLideUtils::ReadConfigFile(const std::string &path, GLideSettings *GPUEngineSettings)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        Logger::LogError("Failed to open configuration file.");
        return false;
    }

    //Direct tracking params
    GPUEngineSettings->directTrackParams.patchSize      = fs["DirTrackParams.patchSize"];
    GPUEngineSettings->directTrackParams.nLevels        = fs["DirTrackParams.nLevels"];
    GPUEngineSettings->directTrackParams.scaleFactor    = fs["DirTrackParams.scaleFactor"];

    GPUEngineSettings->directTrackParams.maxPoints      = fs["DirTrackParams.maxPoints"];
    GPUEngineSettings->directTrackParams.enableAlign    = fs["DirTrackParams.enableAlign"];
    GPUEngineSettings->directTrackParams.searchRadius   = fs["DirTrackParams.searchRadius"];
    GPUEngineSettings->directTrackParams.huberK        = fs["DirTrackParams.huberK"];
    GPUEngineSettings->directTrackParams.epsNorm        = fs["DirTrackParams.epsNorm"];
    GPUEngineSettings->directTrackParams.minPoints        = fs["DirTrackParams.minPoints"];
    GPUEngineSettings->directTrackParams.maxIterations  = fs["DirTrackParams.maxIterations"];

    // vec4 (per-level thresholds)
    GPUEngineSettings->directTrackParams.searchThreshold = readInVector(fs, "DirTrackParams.searchThreshold", 0.0f);
    GPUEngineSettings->directTrackParams.rejectThreshold = readInVector(fs, "DirTrackParams.rejectThreshold", 0.0f);
    GPUEngineSettings->directTrackParams.maxShift        = readInVector(fs, "DirTrackParams.maxShift", 0.0f);


    //read in viewer slamSettings
    GPUEngineSettings->gpuEngineParams.runViewer = fs["Viewer.runViewer"];
    GPUEngineSettings->gpuEngineParams.width = fs["Viewer.width"];
    GPUEngineSettings->gpuEngineParams.height = fs["Viewer.height"];
    GPUEngineSettings->gpuEngineParams.windowFramesTitle = static_cast<std::string>(fs["Viewer.windowFramesTitle"]);
    GPUEngineSettings->gpuEngineParams.windowMapTitle = static_cast<std::string>(fs["Viewer.windowMapTitle"]);
    GPUEngineSettings->gpuEngineParams.scaleFactor = fs["Viewer.scaleFactor"];
    GPUEngineSettings->gpuEngineParams.camMoveFactor = fs["Viewer.mouseMoveFactor"];
    GPUEngineSettings->gpuEngineParams.cameraFollow = fs["Viewer.cameraFollow"];
    GPUEngineSettings->gpuEngineParams.followDistance = fs["Viewer.followDistance"];

    GPUEngineSettings->gpuEngineParams.currentKeyFrameColor = readInVector(fs, "Viewer.currentKeyFrameColor");
    GPUEngineSettings->gpuEngineParams.allKeyFrameColor = readInVector(fs, "Viewer.allKeyFrameColor");
    GPUEngineSettings->gpuEngineParams.tweenFrameDirectColor = readInVector(fs, "Viewer.tweenFrameDirectColor");
    GPUEngineSettings->gpuEngineParams.tweenFrameColor = readInVector(fs, "Viewer.tweenFrameColor");
    GPUEngineSettings->gpuEngineParams.mapPointsColor = readInVector(fs, "Viewer.mapPointsColor");
    GPUEngineSettings->gpuEngineParams.mapPointsRefColor = readInVector(fs, "Viewer.mapPointsRefColor");
    GPUEngineSettings->gpuEngineParams.featureLinesColor = readInVector(fs, "Viewer.featureLinesColor");

    GPUEngineSettings->gpuEngineParams.fov = fs["Viewer.fov"];
    GPUEngineSettings->gpuEngineParams.far = fs["Viewer.far"];
    GPUEngineSettings->gpuEngineParams.near = fs["Viewer.near"];
    GPUEngineSettings->gpuEngineParams.featuresMaxDepth = fs["Viewer.featuresMaxDepth"];
    GPUEngineSettings->gpuEngineParams.forceOriginStart = fs["Viewer.forceOriginStart"];

    return true;
}
