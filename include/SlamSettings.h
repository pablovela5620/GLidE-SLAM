#pragma once
#ifndef SLAM_PARAMS_H
#define SLAM_PARAMS_H

#include <vector>
#include <string>
#include <glm/glm.hpp>

struct SlamSettings
{
    SlamSettings& operator=(const SlamSettings& other)
	{
		if (this != &other)
		{
            viewerParams    = other.viewerParams;
            viewerParams    = other.viewerParams;
			directTrackParams = other.directTrackParams;
		}
		return *this;
	}

	struct ViewerParams
    {
	    int forceOriginStart{1};
	    int runViewer{1};

    	int width{640};
    	int height{480};

    	float scaleFactor{1.0f};
    	int cameraFollow{false};
    	float followDistance{1.0f};

    	float fov{90.0f};
    	float far{1000.0f};
    	float near{1.0f};
    	float camMoveFactor{1.0f};

    	float featuresMaxDepth{10.0f};
    	std::string windowFramesTitle {"GLidE-SLAM: 2D Frames"};
    	std::string windowMapTitle {"GLidE-SLAM: 3D Map"};

    	//gl elements default color set to black
    	glm::vec3 currentKeyFrameColor				{ 0.0f,0.0f,0.0f };
    	glm::vec3 allKeyFrameColor					{ 0.0f,0.0f,0.0f };
    	glm::vec3 tweenFrameDirectColor				{ 0.0f,0.0f,0.0f };
    	glm::vec3 tweenFrameColor					{ 0.0f,0.0f,0.0f };
    	glm::vec3 mapPointsColor					{ 0.0f,0.0f,0.0f };
    	glm::vec3 mapPointsRefColor					{ 0.0f,0.0f,0.0f };
    	glm::vec3 featureLinesColor					{0.0f,0.0f,0.0f};

        float camFov{ 0.0f };
        glm::vec3 camPos   { 0.0f,0.0f,0.0f };
        glm::vec3 camRight { 1.0f,0.0f,0.0f };
        glm::vec3 camUp    { 0.0f,1.0f,0.0f };
        glm::vec3 camTarget{ 0.0f,0.0f,1.0f };
    }viewerParams;

	struct DirectTrackingParams
	{
		int runDirectTracking{1};
		int patchRadius{4};
		int nLevels{4};
		int minUsed{200};
		int trackCamIter{8};
		float maxResidualPx{0.1f};
		float maxDirectError{0.02f};
		float scaleFactor{1.2f};
		float minGrad{0.01};
		float huberDelta{0.03f};
		float kfDisplacement{0.5f};
		int kfMinFrames{3};
	}directTrackParams;

};

#endif // !SLAM_PARAMS_H
