/*
* GPUEngine - GL-accelerated Indirect Direct Embedded SLAM
 *
 * Copyright (c) 2025 [Carlos A. Pinheiro de Sousa / University of Konstanz]
 *
 * This file is part of GLidE-SLAM and is provided under a PROPRIETARY LICENSE.
 * Unlike other parts of this project (licensed under GPL-3.0), this file may NOT be:
 *   - Used in commercial products without written permission
 *   - Redistributed in modified form
 *   - Used to train machine learning models
 *
 * For academic/research use: Free to use with citation.
 * For commercial licensing: Contact [carlos.pinheiro-de-sousa@uni-konstanz.de]
 *
 * If you use this code in academic work, please cite:
 *   [Your IROS 2025 paper citation here]
 */

#include "GLideEngine.h"

bool GLideCompute::initialize()
{
    //TODO: Take all from gpuEngineSettings
    m_width = m_GLideSettings->directTrackParams.width;
    m_height = m_GLideSettings->directTrackParams.height;
    m_nLevels = m_GLideSettings->directTrackParams.nLevels;
    m_patchSize = m_GLideSettings->directTrackParams.patchSize;
    m_patchCenter = static_cast<float>((m_patchSize - 1)) * 0.5f;
    m_patchArea = m_patchSize * m_patchSize;
    m_scaleFactor = m_GLideSettings->directTrackParams.scaleFactor;
    m_fx = m_GLideSettings->directTrackParams.fx;
    m_fy = m_GLideSettings->directTrackParams.fy;
    m_cx = m_GLideSettings->directTrackParams.cx;
    m_cy = m_GLideSettings->directTrackParams.cy;

    m_enableAlign = m_GLideSettings->directTrackParams.enableAlign;
    m_searchRadius = m_GLideSettings->directTrackParams.searchRadius;
    m_humberK = m_GLideSettings->directTrackParams.huberK;

    // Per-level thresholds
    m_searchThreshold = m_GLideSettings->directTrackParams.searchThreshold;
    m_rejectThreshold = m_GLideSettings->directTrackParams.rejectThreshold;
    m_maxShift = m_GLideSettings->directTrackParams.maxShift;

    m_invScaleFactors.resize(m_nLevels);
    m_invScaleFactors[0] = 1.0f;
    for (int i = 1; i < m_nLevels; i++)
    {
        m_invScaleFactors[i] = (m_invScaleFactors[i - 1]/m_scaleFactor);
    }

    bool iniitalizeOk = true;
    if (!initializeImagePyramids())
    {
        Logger::LogError("GPUCompute: Error at initialize ImagePyramids.");
        iniitalizeOk = false;
    }
    if (!initializePreCompute())
    {
        Logger::LogError("GPUCompute: Error at initialize PreCompute.");
        iniitalizeOk = false;
    }
    if (!initializeTrack())
    {
        Logger::LogError("GPUCompute: Error at initialize Track.");
        iniitalizeOk = false;
    }

    return iniitalizeOk;
}

bool GLideCompute::setShaders(const std::map<std::string, std::shared_ptr<Shader> >& shaders)
{
    auto it = shaders.find("convert8UCTo32FShader");
    if (it == shaders.end() || !it->second)
        { Logger::LogError("Failed to load: convert8UCTo32FShader"); return false; }
    GLuint convert8To32FShader = it->second->getHandle();

    it = shaders.find("gauss32FShader");
    if (it == shaders.end() || !it->second)
        { Logger::LogError("Failed to load: gauss32FShader"); return false; }
    GLuint gaussShader32FShader = it->second->getHandle();

    it = shaders.find("resizeShader");
    if (it == shaders.end() || !it->second)
        { Logger::LogError("Failed to load: resizeShader"); return false; }
    GLuint resizeShader = it->second->getHandle();

    it = shaders.find("copyToSSBOShader");
    if (it == shaders.end() || !it->second)
        { Logger::LogError("Failed to load: copyToSSBOShader"); return false; }
    GLuint ssboShader = it->second->getHandle();

    it = shaders.find("preComputeShader");
    if (it == shaders.end() || !it->second)
        { Logger::LogError("Failed to load: preComputeShader"); return false; }
    GLuint preComputeShader = it->second->getHandle();

    it = shaders.find("redPreComputeH1Shader");
    if (it == shaders.end() || !it->second)
        { Logger::LogError("Failed to load: redPreComputeH1Shader"); return false; }
    GLuint redPreComputeH1Shader = it->second->getHandle();

    it = shaders.find("redPreComputeH2Shader");
    if (it == shaders.end() || !it->second)
        { Logger::LogError("Failed to load: redPreComputeH2Shader"); return false; }
    GLuint redPreComputeH2Shader = it->second->getHandle();

    it = shaders.find("trackShader");
    if (it == shaders.end() || !it->second)
        { Logger::LogError("Failed to load: trackShader"); return false; }
    GLuint trackShader = it->second->getHandle();

    it = shaders.find("redTrackShader");
    if (it == shaders.end() || !it->second)
        { Logger::LogError("Failed to load: redTrackShader"); return false; }
    GLuint redTrackShader = it->second->getHandle();

    it = shaders.find("solveTrackShader");
    if (it == shaders.end() || !it->second)
        { Logger::LogError("Failed to load: solveTrackShader"); return false; }
    GLuint solveTrackShader = it->second->getHandle();


    // make sure shader handles loaded
    if (convert8To32FShader == 0 ||
        gaussShader32FShader == 0 ||
        resizeShader == 0 ||
        ssboShader == 0 ||                 // FIXED: == not =
        preComputeShader == 0 ||
        redPreComputeH1Shader == 0 ||
        redPreComputeH2Shader == 0 ||
        trackShader == 0 ||
        redTrackShader == 0 ||
        solveTrackShader == 0)
    {
        return false;
    }

    // set shader handles
    m_convert8UCTo32FShader   = convert8To32FShader;
    m_gauss32FShader          = gaussShader32FShader;
    m_resizeShader            = resizeShader;
    m_copySSBOShader          = ssboShader;
    m_preComputeShader        = preComputeShader;
    m_redH1PreComputeShader   = redPreComputeH1Shader;
    m_redH2PreComputeShader   = redPreComputeH2Shader;
    m_trackShader             = trackShader;
    m_red1TrackShader         = redTrackShader;
    m_solveTrackShader        = solveTrackShader;

    //set shader uniforms (pyramid shader)
    m_uBlurDirPyramid = glGetUniformLocation(m_gauss32FShader, "uDirection");
    m_uScaleFactorPyramid = glGetUniformLocation(m_resizeShader, "uScaleFactor");
    m_copyWidthUniform = glGetUniformLocation(m_copySSBOShader, "uWidth");
    m_uInputTexPyramid = glGetUniformLocation(m_convert8UCTo32FShader, "uInputTexture");

    //set shader uniforms (precompute shader)
    m_uPosePreCompute = glGetUniformLocation(m_preComputeShader, "uPose");
    m_uKPreCompute = glGetUniformLocation(m_preComputeShader, "uK");
    m_uPatchSizePreCompute = glGetUniformLocation(m_preComputeShader, "uPatchSize");
    m_uLevelPreCompute = glGetUniformLocation(m_preComputeShader, "uLevel");
    m_uNpointsPreCompute    = glGetUniformLocation(m_preComputeShader, "uNPoints");
    m_uRefTexPreCompute = glGetUniformLocation(m_preComputeShader, "uRefTexture");
    m_uReduce1PreCompute = glGetUniformLocation(m_redH1PreComputeShader, "uNPoints");
    m_uReduce2PreCompute = glGetUniformLocation(m_redH2PreComputeShader, "uNGroups");

    //set shader uniforms (track shader)
    m_uEnableAlignTrack = glGetUniformLocation(m_trackShader, "uEnableAlign");
    m_uIterationTrack = glGetUniformLocation(m_trackShader, "uIteration");
    m_uPoseTrack = glGetUniformLocation(m_trackShader, "uPose");
    m_uKTrack = glGetUniformLocation(m_trackShader, "uK");
    m_uPatchSizeTrack = glGetUniformLocation(m_trackShader, "uPatchSize");
    m_uLevelTrack = glGetUniformLocation(m_trackShader, "uLevel");
    m_uNewTexTrack = glGetUniformLocation(m_trackShader, "uNewTexture");
    m_uNpointsTrack = glGetUniformLocation(m_trackShader, "uNPoints");
    m_uSearchRadiusTrack = glGetUniformLocation(m_trackShader, "uSearchRadius");
    m_uSearchThresholdTrack = glGetUniformLocation(m_trackShader, "uSearchThreshold");
    m_uRejectThresholdTrack = glGetUniformLocation(m_trackShader, "uRejectThreshold");
    m_uMaxShiftTrack = glGetUniformLocation(m_trackShader, "uMaxShift");
    m_uHumberKTrack = glGetUniformLocation(m_trackShader, "uHuberK");

    //set shader uniforms (reduce track shader)
    m_uNpointsReduce1Track = glGetUniformLocation(m_red1TrackShader, "uNPoints");

    //set shader uniforms (solve track shader)
    m_uPatchSizeSolveTrack = glGetUniformLocation(m_solveTrackShader, "uPatchSize");
    m_uIterationSolveTrack = glGetUniformLocation(m_solveTrackShader, "uIteration");
    m_uMinMeasurementsSolveTrack = glGetUniformLocation(m_solveTrackShader, "uMinMeasurements");
    m_uEpsNormSolveTrack = glGetUniformLocation(m_solveTrackShader, "uEpsNorm");
    m_uIsLastIterationSolveTrack = glGetUniformLocation(m_solveTrackShader, "uIsLastIteration");


    //used for debugging (compare image pyramids)
    // Create readback SSBO (size for largest level)
    glGenBuffers(1, &m_readbackSSBO);

    return (glGetError() == GL_NO_ERROR);
}

bool GLideCompute::initializeImagePyramids()
{
    //initialize level texture dimensions:
    m_levelWidth.resize(m_nLevels);
    m_levelHeight.resize(m_nLevels);
    m_levelWidth[0] = m_width;
    m_levelHeight[0] = m_height;
    for (int i = 1; i < m_nLevels; i++)
    {
        m_levelWidth[i] = floor(((float)m_levelWidth[i - 1] / m_scaleFactor) + 0.5);
        m_levelHeight[i] = floor(((float)m_levelHeight[i - 1] / m_scaleFactor) + 0.5);
    }


    //First texture storage comes in as 8UC (red channel)
    glGenTextures(1, &m_sourceTextureR8);
    glBindTexture(GL_TEXTURE_2D, m_sourceTextureR8);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, m_levelWidth[0], m_levelHeight[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    //initialize and allocate all image pyramid textures storage
    m_pyrTexHandles.resize(m_nLevels);
    glGenTextures(m_nLevels, m_pyrTexHandles.data());


    //because preCompute/track shaders need to perform bilinear-interpolation:
    //sampling/filtering must be set to GL_LINEAR
    for (int L = 0; L < m_nLevels; ++L)
    {
        glBindTexture(GL_TEXTURE_2D, m_pyrTexHandles[L]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, m_levelWidth[L], m_levelHeight[L]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }


    //initialize and allocate temporary and blur textures storage
    m_tempTexHandles.resize(m_nLevels - 1);
    m_blurTexHandles.resize(m_nLevels - 1);
    glGenTextures(m_nLevels - 1, m_tempTexHandles.data());
    glGenTextures(m_nLevels - 1, m_blurTexHandles.data());

    for (int L = 0; L < m_nLevels - 1; ++L)
    {
        glBindTexture(GL_TEXTURE_2D, m_tempTexHandles[L]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, m_levelWidth[L], m_levelHeight[L]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glBindTexture(GL_TEXTURE_2D, m_blurTexHandles[L]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, m_levelWidth[L], m_levelHeight[L]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }


    glBindTexture(GL_TEXTURE_2D, 0);

    return (glGetError() == GL_NO_ERROR);
}

bool GLideCompute::buildPyramid( cv::Mat& image)
{
    //we want explicitly to have 8bit char
    if (image.type() != CV_8UC1) return false;
    if (!image.isContinuous()) image = image.clone();

    if (image.cols != m_levelWidth[0] || image.rows != m_levelHeight[0]) return false;

    if (m_sourceTextureR8 == 0) return false;
    if ((int)m_pyrTexHandles.size() != m_nLevels) return false;
    if ((int)m_tempTexHandles.size() != m_nLevels - 1) return false;
    if ((int)m_blurTexHandles.size() != m_nLevels - 1) return false;

    //Upload first 8 bit uchar texture
    glBindTexture(GL_TEXTURE_2D, m_sourceTextureR8);

    // For first image is 8bit (uchar).
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    glTexSubImage2D(GL_TEXTURE_2D,
                    0,
                    0, 0,
                    m_levelWidth[0], m_levelHeight[0],
                    GL_RED,
                    GL_UNSIGNED_BYTE,
                    image.ptr<uchar>());

#if GLIDE_DEBUG_GL_ERRORS
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) std::cout << "texSubImage err: 0x" << std::hex << err << std::dec << std::endl;
#endif

    glBindTexture(GL_TEXTURE_2D, 0);

    auto ceilDiv = [](int a, int b) -> GLuint { return (GLuint)((a + (b - 1)) / b); };

    // use shader to convert R8 -> R32F into pyramid level 0
    glUseProgram(m_convert8UCTo32FShader);
    glActiveTexture(GL_TEXTURE0); //select texture unit 0
    glBindTexture(GL_TEXTURE_2D, m_sourceTextureR8); //bind the texture to unit 0
    //stores the integer 0 into the sampler uniform, the shader reads from texture unit index 0.
    glUniform1i(m_uInputTexPyramid, 0); //input texture sample from unit 0
    //binds image pyramid [0] as image to image unit 1 (in shader: binding = 1)
    glBindImageTexture(1, m_pyrTexHandles[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    //dispatch compute shader (16, 16, 1 workgroups), threads: (16*16, total threads)
    glDispatchCompute(ceilDiv(m_levelWidth[0], 16), ceilDiv(m_levelHeight[0], 16), 1);

#if GLIDE_DEBUG_GL_ERRORS
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) std::cout << "convert err: 0x" << std::hex << err << std::dec << std::endl;
#endif

    //Finish and commit all image writes done by previous compute work: wait until image is fully written pyramid [0]
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // Build remaining pyramid levels (ALL 32F) from L -1 -> to -> L (level 0 to 1, blur downscale... )
    for (int L = 1; L < m_nLevels; ++L)
    {
        const int srcW = m_levelWidth[L - 1];
        const int srcH = m_levelHeight[L - 1];
        const int dstW = m_levelWidth[L];
        const int dstH = m_levelHeight[L];

        // Gauss blur shader
        // Read from pyramidTexture Handle [L - 1] -> apply blur and write to tempTexture Handle [L-1]
        glUseProgram(m_gauss32FShader);
        glUniform2i(m_uBlurDirPyramid, 0, 1); //set direction to vertical
        glBindImageTexture(0, m_pyrTexHandles[L - 1], 0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F);
        glBindImageTexture(1, m_tempTexHandles[L - 1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glDispatchCompute(ceilDiv(srcW, 16), ceilDiv(srcH, 16), 1);

#if GLIDE_DEBUG_GL_ERRORS
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) std::cout << "L" << L << " gaussV err: 0x" << std::hex << err << std::dec << std::endl;
#endif

        //wait until previouc compute work is done
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // Gauss Horizontal blur
        // Read from tempTexture Handle [L - 1] -> apply blur and write to blurTexture Handle [L-1]
        glUseProgram(m_gauss32FShader);
        glUniform2i(m_uBlurDirPyramid, 1, 0); //set direction to horizontal
        glBindImageTexture(0, m_tempTexHandles[L - 1], 0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F);
        glBindImageTexture(1, m_blurTexHandles[L - 1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glDispatchCompute(ceilDiv(srcW, 16), ceilDiv(srcH, 16), 1);

#if GLIDE_DEBUG_GL_ERRORS
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) std::cout << "L" << L << " gaussH err: 0x" << std::hex << err << std::dec << std::endl;
#endif

        //wait until previouc compute work is done
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        //Resize
        // Read from blurTexture Handle [L - 1] -> apply resize and write to pyramid Texture Handle [L]
        glUseProgram(m_resizeShader);
        glUniform1f(m_uScaleFactorPyramid, m_scaleFactor);
        glBindImageTexture(0, m_blurTexHandles[L - 1], 0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F);
        glBindImageTexture(1, m_pyrTexHandles[L],      0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glDispatchCompute(ceilDiv(dstW, 16), ceilDiv(dstH, 16), 1);

#if GLIDE_DEBUG_GL_ERRORS
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) std::cout << "L" << L << " resize err: 0x" << std::hex << err << std::dec << std::endl;
#endif

        //wait until previouc compute work is done
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    //keeps pyramid-building safe internally, makes the final pyramid textures safe to sample in your preCompute/track shader
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    glUseProgram(0);
    glBindTexture(GL_TEXTURE_2D, 0);

    bool debugDisplay = false;
    if (debugDisplay)
    {
        //TODO: Remove, only for testing how similar to Opencv image pyramids
        std::vector<cv::Mat> m_pyrImg;
        cv::Mat gray32f;
        image.convertTo(gray32f, CV_32FC1, 1.0/255.0);
        m_pyrImg.resize(m_nLevels);
        m_pyrImg[0]    = gray32f;

        //build image pyramids
        for (int L = 1; L < m_nLevels; ++L)
        {
            cv::Mat smoothed;
            cv::GaussianBlur(m_pyrImg[L-1], smoothed, cv::Size(5,5), 1.0, 1.0, cv::BORDER_REPLICATE);
            cv::resize(smoothed, m_pyrImg[L], cv::Size(m_levelWidth[L], m_levelHeight[L]), 0, 0, cv::INTER_LINEAR);
        }

        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        for (int L = 1; L < m_nLevels; ++L)
        {
            int w = m_levelWidth[L];
            int h = m_levelHeight[L];

            cv::Mat gpuLevel = readbackTexture(m_pyrTexHandles[L], w, h);

            double gpuMin, gpuMax;
            cv::minMaxLoc(gpuLevel, &gpuMin, &gpuMax);
            std::cout << "Level " << L << " (" << w << "x" << h << ") gpuMin=" << gpuMin << " gpuMax=" << gpuMax;

            if (w == m_pyrImg[L].cols && h == m_pyrImg[L].rows)
            {
                cv::Mat diff;
                cv::absdiff(m_pyrImg[L], gpuLevel, diff);
                double maxVal;
                cv::minMaxLoc(diff, nullptr, &maxVal);
                std::cout << " maxDiff=" << maxVal << " mean=" << cv::mean(diff)[0];
            }
            std::cout << std::endl;


            if (w == m_pyrImg[L].cols && h == m_pyrImg[L].rows)
            {
                cv::Mat diff;
                cv::absdiff(m_pyrImg[L], gpuLevel, diff);

                double minVal, maxVal;
                cv::minMaxLoc(diff, &minVal, &maxVal);
                std::cout << "L" << L << " maxDiff=" << maxVal << std::endl;

                // Normalize to full 0-255 range so differences are visible
                cv::Mat diffVis;
                if (maxVal > 0.0)
                    diff.convertTo(diffVis, CV_8U, 255.0 / maxVal);
                else
                    diffVis = cv::Mat::zeros(diff.size(), CV_8U);

                cv::imshow("Diff L" + std::to_string(L), diffVis);
            }
        }

        cv::waitKey(0);
    }
    return true;
}

bool GLideCompute::initializePreCompute()
{
    // This function generates pre-allocates buffers.
    // certain parameters are set based a capped max. n. of points
    // this avoids allocating new buffers every frame.

    //Input buffer: map points allocate space for max. n of points
    glGenBuffers(1, &m_ssboMapPoints);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboMapPoints);
    glBufferData(GL_SHADER_STORAGE_BUFFER,  (GLsizeiptr)(m_maxPoints * sizeof(glm::vec4)), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,0);
    if (glGetError() != GL_NO_ERROR) return false;


    //Output: One per level for every output of cache (SSBOs).
    m_preComputeCache.resize(m_nLevels);

    for (int L = 0; L < m_nLevels; ++L)
    {
        auto& cacheLevel = m_preComputeCache[L];

        //generate buffers: 1 buffer object, store it in named cache at level L
        glGenBuffers(1, &cacheLevel.ssbo_isValid);
        glGenBuffers(1, &cacheLevel.ssbo_I);
        glGenBuffers(1, &cacheLevel.ssbo_J);
        glGenBuffers(1, &cacheLevel.ssbo_H);

        glGenBuffers(1,&cacheLevel.ssbo_HLevel);

        //check if any issues
        if (cacheLevel.ssbo_isValid == 0
            || cacheLevel.ssbo_I == 0
            || cacheLevel.ssbo_J == 0
            || cacheLevel.ssbo_H == 0
            || cacheLevel.ssbo_HLevel == 0)
        {
            Logger::LogError("Error at SSBOs generation; initializePreCompute.");
            return false;
        }


        //Buffers allocation used in shader preComputeShader
        //Point valid or not (size of map points)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_isValid);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(m_maxPoints * sizeof(uint32_t)), nullptr, GL_DYNAMIC_DRAW);

        //Patch intensities (size of map points *  patch area, i.e. number of pixels in patch)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_I);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(m_maxPoints * (uint32_t)m_patchArea * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);

        //Jacobians (each pixel contributes 6 values, wx,wy,wz,tx,ty,tz (camera pose) per patch for every point
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_J);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(m_maxPoints * (uint32_t)m_patchArea * 6u * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);

        //Hessians 6 x 6 (J^T*J), upper triangle from matrix, 21 values) per point
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_H);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(m_maxPoints * 21u * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);


        //reduction shader buffers
        //final level are 21 values (half-Hessian)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_HLevel);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(21u * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return (glGetError() == GL_NO_ERROR);

}

bool GLideCompute::preCompute(const std::vector<glm::vec4> &mapPoints, const cv::Mat &pose)
{
    if (m_ssboMapPoints == 0) return false;
    if (mapPoints.empty() || mapPoints.size() > m_maxPoints) return false;
    if (m_nLevels == 0) return false;

    m_nPoints = mapPoints.size();

    //load preCompute shader
    glUseProgram(m_preComputeShader);

    //only update map points data (glBufferSubData)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssboMapPoints);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER,
        0,
        m_nPoints * sizeof(glm::vec4),
        mapPoints.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    //connects buffer object to SSBO indexed binding slot
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssboMapPoints);


    //convert pose from opencv -> glm (glsl)
    glm::mat4 glmPose(1.0f);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            glmPose[j][i] = pose.at<float>(i, j);

    //write to shader uniforms
    //uLevel and intrinsics uniforms are level-dependent, so they are set in loop
    glUniformMatrix4fv(m_uPosePreCompute, 1, GL_FALSE, &glmPose[0][0]);
    glUniform1i(m_uPatchSizePreCompute, m_patchSize);
    glUniform1i(m_uNpointsPreCompute, (GLint)m_nPoints);
    glUniform1i(m_uRefTexPreCompute,0); //input texture sample from unit 0

    //main loop for precompute inverse-compositional
    //course-to-fine here is actually irrelevant
    for (int L = m_nLevels - 1; L >= 0; --L)
    {
        //Inputs:
        //update level uniform
        glUniform1i(m_uLevelPreCompute, L);

        // update intrinsics (pre-scaled) uniforms
        float fx = m_fx * m_invScaleFactors[L];
        float fy = m_fy * m_invScaleFactors[L];
        float cx = m_cx * m_invScaleFactors[L];
        float cy = m_cy * m_invScaleFactors[L];
        glUniform4f(m_uKPreCompute, fx , fy, cx, cy);

        //bind image pyramid level
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_pyrTexHandles[L]);

        auto& cacheLevel = m_preComputeCache[L];

        //connect buffer object to SSBO indexed binding slot(type of storage, slot number, buffer to access)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cacheLevel.ssbo_isValid);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cacheLevel.ssbo_I);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cacheLevel.ssbo_J);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cacheLevel.ssbo_H);

        //launch precompute shader
        glDispatchCompute((m_nPoints + 63) / 64, 1, 1);

        //wait for completion
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

        //clearPreComputeReduction(L);

        //Second-phase: Reduce H:
        //Sum per-point partial
        glUseProgram(m_redH1PreComputeShader);

        glUniform1ui(m_uReduce1PreCompute, (GLint)m_nPoints);

        //connect buffer object to SSBO indexed binding slot(type of storage, slot number, buffer to access)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cacheLevel.ssbo_isValid);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cacheLevel.ssbo_H);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, cacheLevel.ssbo_HLevel);

        //launch 1st pass reduction shader
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        glUseProgram(0);

        // switch back for next level's precompute
        glUseProgram(m_preComputeShader);

    }


    glUseProgram(0);
    glBindTexture(GL_TEXTURE_2D, 0);

#if GLIDE_DEBUG_GL_ERRORS
    if (glGetError() != GL_NO_ERROR) return false;
    glFinish();
#endif

    return true;
}

bool GLideCompute::initializeTrack()
{

    m_trackCache.resize(m_nLevels);

    glGenBuffers(1, &m_ssbo_PoseTrack);
    if (m_ssbo_PoseTrack == 0)
    {
        Logger::LogError("Error at SSBOs generation; initializeTrack.");
        return false;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_PoseTrack);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(sizeof(glm::mat4)), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    //Allocate buffers, 1 per level
    for (size_t i = 0; i < m_nLevels; ++i)
    {
        auto& cacheLevel = m_trackCache[i];

        //generate buffers: 1 buffer object, store it in named cache at level L
        glGenBuffers(1, &cacheLevel.ssbo_B0);
        glGenBuffers(1, &cacheLevel.ssbo_B1);
        glGenBuffers(1,&cacheLevel.ssbo_Chi2);
        glGenBuffers(1,&cacheLevel.ssbo_isValid);
        glGenBuffers(1,&cacheLevel.ssbo_Align);
        glGenBuffers(1,&cacheLevel.ssbo_State);

        glGenBuffers(1,&cacheLevel.ssbo_B0Level);
        glGenBuffers(1,&cacheLevel.ssbo_B1Level);
        glGenBuffers(1,&cacheLevel.ssbo_Chi2Level);
        glGenBuffers(1,&cacheLevel.ssbo_isValidLevel);

        //check if any issues
        if (cacheLevel.ssbo_B0 == 0
            || cacheLevel.ssbo_B1 == 0
            || cacheLevel.ssbo_Chi2 == 0
            || cacheLevel.ssbo_isValid == 0
            || cacheLevel.ssbo_Align == 0
            || cacheLevel.ssbo_State == 0

            || cacheLevel.ssbo_B0Level == 0
            || cacheLevel.ssbo_B1Level == 0
            || cacheLevel.ssbo_Chi2Level == 0
            || cacheLevel.ssbo_isValidLevel == 0)
        {
            Logger::LogError("Error at SSBOs generation; initializeTrack.");
            return false;
        }

        //Buffers allocation used in shader trackShader
        //b output, vec4 first 4 elements (b0,b1,b2,b3)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_B0);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(m_maxPoints * sizeof(glm::vec4)), nullptr, GL_DYNAMIC_DRAW);

        //b output, vec4 last 2 elements (b4,b5,0,0)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_B1);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(m_maxPoints * sizeof(glm::vec4)), nullptr, GL_DYNAMIC_DRAW);

        //Chi2 output, floats
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_Chi2);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(m_maxPoints * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);

        //isValid, uint 1 or 0
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_isValid);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(m_maxPoints * sizeof(uint32_t)), nullptr, GL_DYNAMIC_DRAW);

        //Align, vec4 (keeps du,dv, valid) for each point
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_Align);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(m_maxPoints * sizeof(glm::vec4)), nullptr, GL_DYNAMIC_DRAW);        //Align, vec4 (keeps du,dv, valid) for each point

        //State block several state variables used during iterations/levels
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_State);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(sizeof(TrackStateBlock)), nullptr, GL_DYNAMIC_DRAW);


        //buffer allocation used in reduction shader
        //b output, vec4 first 4 elements (b0,b1,b2,b3)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_B0Level);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(sizeof(glm::vec4)), nullptr, GL_DYNAMIC_DRAW);

        //b output, vec4 last 2 elements (b4,b5,0,0)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_B1Level);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(sizeof(glm::vec4)), nullptr, GL_DYNAMIC_DRAW);

        //Chi2 output, floats
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_Chi2Level);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);

        //isValid, uint 1 or 0
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_isValidLevel);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(sizeof(uint32_t)), nullptr, GL_DYNAMIC_DRAW);

    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return (glGetError() == GL_NO_ERROR);
}

bool GLideCompute::track(uint32_t frameID, cv::Mat& pose, float &outChi2, int &outN)
{
    if (pose.empty() || pose.type() != CV_32FC1 || m_nLevels <= 0 || m_nPoints == 0 || m_nPoints > m_maxPoints)
    {
        std::lock_guard<std::mutex> lock(m_gpuTrackResult.mutex);
        m_gpuTrackResult.frameID = frameID;
        m_gpuTrackResult.pose = cv::Mat();
        m_gpuTrackResult.chi2 = 0.0f;
        m_gpuTrackResult.N = 0;
        m_gpuTrackResult.success = false;
        m_gpuTrackResult.ready = true;
        m_gpuTrackResult.resultReady.notify_one();
        return false;
    }

    auto publishFail = [&]()
    {
        {
            std::lock_guard<std::mutex> lock(m_gpuTrackResult.mutex);
            m_gpuTrackResult.frameID = frameID;
            m_gpuTrackResult.pose = cv::Mat();
            m_gpuTrackResult.chi2 = 0.0f;
            m_gpuTrackResult.N = 0;
            m_gpuTrackResult.success = false;
            m_gpuTrackResult.ready = true;
        }
        m_gpuTrackResult.resultReady.notify_one();
    };

    const uint32_t lastIteration = m_maxIterations - 1;

    //start
    cv::Mat Tcw = pose.clone();

    //copy pose to buffer
    float m[16];
    for (size_t r = 0; r < 4;++r)
        for (size_t c = 0; c < 4; ++c)
            m[c*4 + r] = Tcw.at<float>(r, c);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_PoseTrack);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(m), m);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, TRACK_INOUT_POSE, m_ssbo_PoseTrack);


    //Track state cache needs to be cleared at each call of track (all levels)
    for (int L = m_nLevels-1; L >= 0; --L)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_trackCache[L].ssbo_State);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(TrackStateBlock), &m_trackSateData);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    //write to shader uniforms
    //uLevel and uK uniforms are level-dependent, so they are set in loop
    //uIteration is per iteration dependent, set in iteration loop

    outChi2 = 0.0f;
    outN = 0;

    float finalChi2Mean = std::numeric_limits<float>::max();
    bool anyLevelOk = false;


    glUseProgram(m_trackShader);
    //uniforms independent of iteration/Level
    glUniform1i(m_uNpointsTrack, (GLint)m_nPoints);

    glUniform1i(m_uPatchSizeTrack, m_patchSize);
    glUniform1ui(m_uEnableAlignTrack, m_enableAlign);
    glUniform1ui(m_uSearchRadiusTrack, (GLuint)m_searchRadius);
    glUniform4fv(m_uSearchThresholdTrack, 1, &m_searchThreshold[0]);
    glUniform4fv(m_uRejectThresholdTrack, 1, &m_rejectThreshold[0]);
    glUniform4fv(m_uMaxShiftTrack, 1, &m_maxShift[0]);
    glUniform1f(m_uHumberKTrack, m_humberK);


    //Main loop, course to fine levels
    for (int L = m_nLevels-1; L >= 0; --L)
    {
        for (uint32_t iteration = 0; iteration < m_maxIterations; ++iteration)
        {
            //************************************ TRACK SHADER ************************************
            //uniforms per iteration
            glUniform1i(m_uLevelTrack, L);
            glUniform1i(m_uIterationTrack, iteration);

            // update intrinsics (pre-scaled) uniforms
            float fx = m_fx * m_invScaleFactors[L];
            float fy = m_fy * m_invScaleFactors[L];
            float cx = m_cx * m_invScaleFactors[L];
            float cy = m_cy * m_invScaleFactors[L];
            glUniform4f(m_uKTrack, fx , fy, cx, cy);


            //bind current image pyramid as sampler
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D,m_pyrTexHandles[L]);
            glUniform1i(m_uNewTexTrack,0);


            //BIND SSBOs for trackShader:
            //READ-ONLY
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,TRACK_IN_MAPPOINTS,m_ssboMapPoints);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,TRACK_IN_VALID,m_preComputeCache[L].ssbo_isValid);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,TRACK_IN_I,m_preComputeCache[L].ssbo_I);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,TRACK_IN_J,m_preComputeCache[L].ssbo_J);

            //WRITE
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,TRACK_OUT_B0,m_trackCache[L].ssbo_B0);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,TRACK_OUT_B1,m_trackCache[L].ssbo_B1);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,TRACK_OUT_CHI2,m_trackCache[L].ssbo_Chi2);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,TRACK_OUT_ISVALID,m_trackCache[L].ssbo_isValid);

            //READ/WRITE
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,TRACK_INOUT_ALIGN,m_trackCache[L].ssbo_Align);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,TRACK_INOUT_POSE,m_ssbo_PoseTrack);


            //Dispatch
            glDispatchCompute((GLuint)((m_nPoints + 63u) / 64u), 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            //************************************ REDUCTION SHADER ************************************
            //clearTrackReduction(L);

            glUseProgram(m_red1TrackShader);
            glUniform1ui(m_uNpointsReduce1Track, (GLuint)m_nPoints);

            //BIND SSBOs for reduce1TrackShader:
            //READ-ONLY
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, REDUCE_IN_B0,      m_trackCache[L].ssbo_B0);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, REDUCE_IN_B1,      m_trackCache[L].ssbo_B1);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, REDUCE_IN_CHI2,    m_trackCache[L].ssbo_Chi2);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, REDUCE_IN_ISVALID, m_trackCache[L].ssbo_isValid);

            //READ-WRITE
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, REDUCE_OUT_B0,      m_trackCache[L].ssbo_B0Level);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, REDUCE_OUT_B1,      m_trackCache[L].ssbo_B1Level);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, REDUCE_OUT_CHI2,    m_trackCache[L].ssbo_Chi2Level);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, REDUCE_OUT_ISVALID, m_trackCache[L].ssbo_isValidLevel);

            glDispatchCompute(1,1,1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);



            //************************************ SOLVE SHADER ************************************
            glUseProgram(m_solveTrackShader);

            glUniform1i(m_uPatchSizeSolveTrack, (GLint)m_patchSize);
            glUniform1i(m_uIterationSolveTrack, iteration);
            glUniform1i(m_uMinMeasurementsSolveTrack, (GLint)m_minMeasurements);
            glUniform1i(m_uIsLastIterationSolveTrack, (iteration == lastIteration) ? 1 : 0);
            glUniform1f(m_uEpsNormSolveTrack, m_epsNorm);

            //READ
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,SOLVE_IN_B0LEVEL,m_trackCache[L].ssbo_B0Level);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,SOLVE_IN_B1LEVEL,m_trackCache[L].ssbo_B1Level);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,SOLVE_IN_CHI2LEVEL,m_trackCache[L].ssbo_Chi2Level);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,SOLVE_IN_VALIDLEVEL,m_trackCache[L].ssbo_isValidLevel);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,SOLVE_IN_HLEVEL,m_preComputeCache[L].ssbo_HLevel);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,SOLVE_INOUT_STATE,m_trackCache[L].ssbo_State);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,SOLVE_INOUT_POSE,m_ssbo_PoseTrack);



            glDispatchCompute(1,1,1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


            //Make sure to set the track shader again for next iteration
            glUseProgram(m_trackShader);
        }
    }

    glUseProgram(0);

    float poseResult[16];
    if (!readSSBO(m_ssbo_PoseTrack, &poseResult, sizeof(poseResult)))
    {
        Logger::LogError("Could not read pose. Aborting.");
        publishFail();
        return false;
    }

    TrackStateBlock trackStateResult;
    if (!readSSBO(m_trackCache[0].ssbo_State, &trackStateResult, sizeof(trackStateResult)))
    {
        Logger::LogError("Could not read track results. Aborting.");
        publishFail();
        return false;
    }

    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            Tcw.at<float>(r,c) = poseResult[c*4 + r];

    pose = Tcw.clone();
    outChi2 = trackStateResult.bestChi;
    outN = int(trackStateResult.bestValidPts) * m_patchArea;
    anyLevelOk = (trackStateResult.failed == 0u) && (trackStateResult.hadValid != 0u);

    {
        std::lock_guard<std::mutex> lock(m_gpuTrackResult.mutex);
        m_gpuTrackResult.frameID = frameID;
        m_gpuTrackResult.pose = Tcw.clone();
        m_gpuTrackResult.chi2 = outChi2;
        m_gpuTrackResult.N = outN;
        m_gpuTrackResult.success = anyLevelOk;
        m_gpuTrackResult.ready = true;
    }
    m_gpuTrackResult.resultReady.notify_one();

    return anyLevelOk;
}

bool GLideCompute::readSSBO(GLuint ssbo, void* destination,size_t numBytes)
{
    //read ssbo
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    void* ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,(GLsizeiptr)numBytes,GL_MAP_READ_BIT);
    if (!ptr)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return false;
    }
    std::memcpy(destination, ptr, numBytes);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

#if GLIDE_DEBUG_GL_ERRORS
    if (glGetError() != GL_NO_ERROR) return false;
#endif
    return true;
}

bool GLideCompute::rebuildH(Eigen::Matrix<float, 6, 6> &H, const float* hTemp)
{
    if (!hTemp) return false;
    auto matrixTriangleIndex = [] (int a, int b)->int{int base = (a*6)-((a*(a-1))/2); return base + (b-a);};

    H = Eigen::Matrix<float,6,6>::Zero();
    for (size_t a = 0; a < 6; ++a)
    {
        for (size_t b = a; b < 6; ++b)
        {
            float value = hTemp[matrixTriangleIndex(a,b)];
            H(a,b) = value;
            H(b,a) = value;
        }
    }

    return (!H.isZero());
}

cv::Matx44f GLideCompute::se3exp(const cv::Matx<float, 6, 1> &xi)
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

bool GLideCompute::shutDown()
{
   Logger::LogInfoI("GPUCompute: Shutting down.");

    glFinish();

    //cleanup map points buffer
    if (m_ssboMapPoints)
    {
        glDeleteBuffers(1, &m_ssboMapPoints);
        m_ssboMapPoints = 0;
    }

    //cleanup preCompute cache buffers
    for (size_t L = 0; L < m_preComputeCache.size(); ++L)
    {
        PreComputeCache& c = m_preComputeCache[L];

        if (c.ssbo_isValid) { glDeleteBuffers(1, &c.ssbo_isValid); c.ssbo_isValid = 0; }
        if (c.ssbo_I)       { glDeleteBuffers(1, &c.ssbo_I);       c.ssbo_I       = 0; }
        if (c.ssbo_J)       { glDeleteBuffers(1, &c.ssbo_J);       c.ssbo_J       = 0; }
        if (c.ssbo_H)       { glDeleteBuffers(1, &c.ssbo_H);       c.ssbo_H       = 0; }
        if (c.ssbo_HLevel)  { glDeleteBuffers(1, &c.ssbo_HLevel);  c.ssbo_HLevel  = 0; }
    }
    m_preComputeCache.clear();

    //cleanup track cache buffers
    for (size_t L = 0; L < m_trackCache.size(); ++L)
    {
        TrackCache& c = m_trackCache[L];

        if (c.ssbo_B0)          { glDeleteBuffers(1, &c.ssbo_B0);          c.ssbo_B0 = 0; }
        if (c.ssbo_B1)          { glDeleteBuffers(1, &c.ssbo_B1);          c.ssbo_B1 = 0; }
        if (c.ssbo_Chi2)        { glDeleteBuffers(1, &c.ssbo_Chi2);        c.ssbo_Chi2 = 0; }
        if (c.ssbo_isValid)     { glDeleteBuffers(1, &c.ssbo_isValid);     c.ssbo_isValid = 0; }
        if (c.ssbo_Align)       { glDeleteBuffers(1, &c.ssbo_Align);       c.ssbo_Align = 0; }

        if (c.ssbo_B0Level)     { glDeleteBuffers(1, &c.ssbo_B0Level);     c.ssbo_B0Level = 0; }
        if (c.ssbo_B1Level)     { glDeleteBuffers(1, &c.ssbo_B1Level);     c.ssbo_B1Level = 0; }
        if (c.ssbo_Chi2Level)   { glDeleteBuffers(1, &c.ssbo_Chi2Level);   c.ssbo_Chi2Level = 0; }
        if (c.ssbo_isValidLevel){ glDeleteBuffers(1, &c.ssbo_isValidLevel);c.ssbo_isValidLevel = 0; }
    }
    m_trackCache.clear();


    //cleanup textures
    if (!m_pyrTexHandles.empty())
    {
        glDeleteTextures((GLsizei)m_pyrTexHandles.size(), m_pyrTexHandles.data());
        m_pyrTexHandles.clear();
    }
    if (!m_tempTexHandles.empty())
    {
        glDeleteTextures((GLsizei)m_tempTexHandles.size(), m_tempTexHandles.data());
        m_tempTexHandles.clear();
    }
    if (!m_blurTexHandles.empty())
    {
        glDeleteTextures((GLsizei)m_blurTexHandles.size(), m_blurTexHandles.data());
        m_blurTexHandles.clear();
    }

    if (m_sourceTextureR8)
    {
        glDeleteTextures(1, &m_sourceTextureR8);
        m_sourceTextureR8 = 0;
    }

    if (m_readbackSSBO)
    {
        glDeleteBuffers(1, &m_readbackSSBO);
        m_readbackSSBO = 0;
    }

    return (glGetError() == GL_NO_ERROR);
}

void GLideCompute::clearTrackReduction(const int Level)
{
    glm::vec4 z4(0,0,0,0);
    float zf = 0.0f;
    uint32_t zu = 0u;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_trackCache[Level].ssbo_B0Level);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(glm::vec4), &z4);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_trackCache[Level].ssbo_B1Level);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(glm::vec4), &z4);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_trackCache[Level].ssbo_Chi2Level);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float), &zf);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_trackCache[Level].ssbo_isValidLevel);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zu);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GLideCompute::clearPreComputeReduction(const int Level)
{
    auto& cacheLevel = m_preComputeCache[Level];
    float zero21[21] = {0};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cacheLevel.ssbo_HLevel);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(zero21), zero21);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

cv::Mat GLideCompute::readbackTexture(GLuint texHandle, int w, int h)
{
    size_t size = (size_t)w * (size_t)h * sizeof(float);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_readbackSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, nullptr, GL_STREAM_READ);

    // IMPORTANT: shader uses layout(std430, binding = 1)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_readbackSSBO);

    glUseProgram(m_copySSBOShader);
    glUniform1i(m_copyWidthUniform, w);
    glBindImageTexture(0, texHandle, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);

    auto ceilDiv = [](int a, int b) { return (a + b - 1) / b; };
    glDispatchCompute(ceilDiv(w, 16), ceilDiv(h, 16), 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

#if GLIDE_DEBUG_GL_ERRORS
    glFinish();
#endif

    cv::Mat result(h, w, CV_32F, cv::Scalar(0));

    void* ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, size, GL_MAP_READ_BIT);
    if (ptr)
    {
        memcpy(result.data, ptr, size);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    else
    {
        GLenum err = glGetError();
        std::cout << "glMapBufferRange failed, err=0x" << std::hex << err << std::dec << std::endl;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);

    return result;
}

bool GLideCompute::getTrackResult(uint32_t frameID, cv::Mat& pose, float& chi2, int& N)
{
    std::unique_lock<std::mutex> lock(m_gpuTrackResult.mutex);

    bool ready = m_gpuTrackResult.resultReady.wait_for(lock,
            std::chrono::milliseconds(10), [&]()
            { return m_gpuTrackResult.ready && (m_gpuTrackResult.frameID == frameID); });


    //
    // if (!m_gpuTrackResult.ready)
    // {
    //     Logger::LogWarning("GPUCompute: track result not ready! dispatched id="
    //         + std::to_string(m_gpuTrackResult.frameID) + " check result id=" + std::to_string(frameID));
    //     return false;
    // }
    // if (m_gpuTrackResult.frameID != frameID)
    // {
    //     Logger::LogWarning("GPUCompute: Mismatch in frame number! dispatched id="
    //         + std::to_string(m_gpuTrackResult.frameID) + " check result id=" + std::to_string(frameID));
    //     return false;
    // }


    pose = m_gpuTrackResult.pose.clone();
    chi2 = m_gpuTrackResult.chi2;
    N = m_gpuTrackResult.N;
    bool success = m_gpuTrackResult.success;
    m_gpuTrackResult.ready = false;

    return success;
}

bool GLideEngine::getTrackResult(uint32_t frameID, cv::Mat& pose, float& chi2, int& N)
{
    if (!m_gpuCompute) return false;
    return m_gpuCompute->getTrackResult(frameID, pose, chi2, N);
}

bool GLideEngine::initialize()
{

    m_isInitialized = true;
    m_width = m_GPUEngineSettings->gpuEngineParams.width;
    m_height = m_GPUEngineSettings->gpuEngineParams.height;

    m_windowFramesTitle = m_GPUEngineSettings->gpuEngineParams.windowFramesTitle;
    m_windowMapTitle = m_GPUEngineSettings->gpuEngineParams.windowMapTitle;

    //color
    m_currentKeyFrameColor = m_GPUEngineSettings->gpuEngineParams.currentKeyFrameColor;
    m_AllKeyFrameColor = m_GPUEngineSettings->gpuEngineParams.allKeyFrameColor;
    m_tweenFrameDirectColor = m_GPUEngineSettings->gpuEngineParams.tweenFrameDirectColor;
    m_tweenFrameColor = m_GPUEngineSettings->gpuEngineParams.tweenFrameColor;
    m_mapPointsColor = m_GPUEngineSettings->gpuEngineParams.mapPointsColor;
    m_mapPointsRefColor = m_GPUEngineSettings->gpuEngineParams.mapPointsRefColor;
    m_featureLinesColor = m_GPUEngineSettings->gpuEngineParams.featureLinesColor;

    m_scaleFactor = m_GPUEngineSettings->gpuEngineParams.scaleFactor;
    m_featuresMaxDepth = m_GPUEngineSettings->gpuEngineParams.featuresMaxDepth;

    m_featuresMaxDepth *= m_scaleFactor;
    m_logTiming = m_GPUEngineSettings->gpuEngineParams.logTiming;

    initializeWindows();
    if (m_windowFrames2D == nullptr)
    {
        Logger::LogError("GPUEngine: Failed to initialize m_windowFrames2D window.");
        m_isInitialized = false;
    }

    if (m_windowMap3D == nullptr)
    {
        Logger::LogError("GPUEngine: Failed to initialize m_windowFrames2D window.");
        m_isInitialized = false;
    }


    initializeCamera();

    //TODO: put in another function, not so clean here
    //initialize current KF frame:
    glm::mat4 pose(1.0f);
    m_currentKeyFrameGfx = new FrameGizmo(0, pose, 0);
    m_currentKeyFrameGfx->initialize();

    //initialize GPUCompute
    m_gpuCompute = new GLideCompute(m_GPUEngineSettings);

    bool gpuComputeOk = true;
    if (!m_gpuCompute->initialize())
    {
        Logger::LogError("GPUEngine: Failed to initialize GPUCompute.");
        gpuComputeOk = false;
    }

    if (!m_gpuCompute->setShaders(m_shaders))
    {
        Logger::LogError("GPUEngine: Failed to set shaders.");
        gpuComputeOk = false;
    }
    if (!gpuComputeOk)
    {
        m_gpuCompute->shutDown();
        m_isInitialized = false;
    }

    if (m_isInitialized)
    {
        Logger::LogInfoI("GPUEngine: initialized.");
    }
    return m_isInitialized;
}

void GLideEngine::run()
{
    if (!m_isInitialized)
    {
        if (!initialize())
        {
            //abort
            Logger::LogInfoI("GPUEngine: failed to initialize, aborting.");
            m_stop.store(false);
        }
    }


    auto loopStart = std::chrono::high_resolution_clock::now();
    int loopCount = 0;

    while(!m_stop.load())
    {
        m_newTime = static_cast<float>(SDL_GetTicks())/1000.0f;
        float dt = [this](float newT, float& oldT)->float{float deltaT = newT - oldT; if(oldT == 0.0f) deltaT = 0.0f; oldT = newT; return deltaT; }(m_newTime, m_oldTime);

        m_activeCamera->update(dt);

        //avoid CPU-GPU transfer every frame
          uint32_t mapPointsUpdateNumber = m_map->GetMapPointsUpdateNumber();
          if (ma_LastMapPointUpdateNumber != mapPointsUpdateNumber)
          {
              ma_LastMapPointUpdateNumber = mapPointsUpdateNumber;
              updateMapPoints();
          }

          uint32_t framesUpdateNumber = m_map->GetFramesUpdateNumber();
          if (ma_LastFramesUpdateNumber != framesUpdateNumber)
          {
              ma_LastFramesUpdateNumber = framesUpdateNumber;
              updateFrames3D();
          }

          // if(checkUpdateFramesFlag())
          // {
          //     //updateDirectMapping();
          // }

        updateDirectTracking();

        render();

        //get framerate (this is from viewer only!)
        float avgFPS = ViewerUtil::getFPS(m_frameTimes,dt,m_N);
        //TODO: TEST WITH SDL_DELAY OR NOT, CURRENTLY REMOVED!
        //SDL_Delay(33);
    }
}

void GLideEngine::updateDirectTracking()
{
    uint32_t frameID = 0;
    cv::Mat img, pose;
    std::vector<glm::vec4> pts;
    bool doPrecompute = false;
    bool doTrack = false;
    float outB[6] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,};
    float outChi2 = 0.0f;
    int outN = 0;

    //use buffer copies (safety):
    //1-> producer writes to m_sourceImage (deep copy in update functionss using .clone()) -> buffer A
    //2-> consumer does shallow copy img = m_sourceImage, shares buffer A (ref count)
    //3-> if producer overwrites m_sourceImage (clone()), m_sourceImage points to buffer B
    //while img still keeps buffer A alive.
    {
        std::lock_guard<std::mutex> lock(m_directTrackingMutex);
        if (m_directTrackDataAvailable && m_gpuCompute != nullptr)
        {

            img = m_sourceImage; //img points to specific address
            pose = m_initialPose;
            frameID = m_sourceFrameID;
            //decide to run precompute on Ref frame or direct tracking in new frame
            if (m_runPrecompute)
            {
                pts.swap(m_slamMapPoints);
                m_runPrecompute = false;
                doPrecompute = true;
                doTrack = false;
            }
            else
            {
                doPrecompute = false;
                doTrack = true;
            }
            m_directTrackDataAvailable = false;
        }
    }

    m_newFrameReady.notify_all();


    //first step (either precompute/direct tracking, build image pyramids)
    if (!img.empty())
    {
        //for normal cases
        if (!m_logTiming)
            m_gpuCompute->buildPyramid(img);
        else //otherwise debug timings to file
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            m_gpuCompute->buildPyramid(img);
            glFinish();
            auto t1 = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float, std::milli>(t1 - t0).count();
            std::string fileContent = "imagePyramid," + std::to_string(frameID) + "," + std::to_string(dt);
            logTiming(fileContent);
        }


    }

    if (doPrecompute)
    {
        //for normal cases
        if (!m_logTiming)
            m_gpuCompute->preCompute(pts, pose);
        else //otherwise debug timings to file
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            m_gpuCompute->preCompute(pts, pose);
            glFinish();
            auto t1 = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float, std::milli>(t1 - t0).count();
            std::string fileContent = "preCompute," + std::to_string(frameID) + "," + std::to_string(dt);
            logTiming(fileContent);
        }

    }
    else if (doTrack)
    {
        //for normal cases
        if (!m_logTiming)
            m_gpuCompute->track(frameID, pose,outChi2,outN);
        else //otherwise debug timings to file
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            bool ok = m_gpuCompute->track(frameID, pose,outChi2,outN);

            auto t1 = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float, std::milli>(t1 - t0).count();
            std::string fileContent =
                "track," + std::to_string(frameID) + "," + std::to_string(dt) + "," +
                std::to_string(outChi2) + "," + (ok ? "1" : "0");
            logTiming(fileContent);
        }
    }
}

void GLideEngine::updateNewFrame(uint32_t frameID, const cv::Mat &image, const cv::Mat &pose)
{
    if (!m_isInitialized || m_gpuCompute== nullptr)
        return;

    {
        std::unique_lock<std::mutex> lock(m_directTrackingMutex);

        m_newFrameReady.wait(lock, [&]()
            {return (!m_directTrackDataAvailable && !m_runPrecompute) || m_stop.load();});

        // //avoid interrupt precompute (this should not happen anyway)
        // if (m_runPrecompute)
        //     return;

        m_sourceFrameID = frameID;
        m_sourceImage = image.clone();
        m_initialPose = pose.clone();
        m_directTrackDataAvailable = true;
    }
}

void GLideEngine::updateRefFrame(const cv::Mat& image, std::vector<glm::vec4> mapPoints,const cv::Mat& pose)
{
    if (!m_isInitialized || m_gpuCompute== nullptr)
        return;
    {
        std::unique_lock<std::mutex> lock(m_directTrackingMutex);

        // wait until slot is free
        m_newFrameReady.wait(lock, [&]()
            {return (!m_directTrackDataAvailable) || m_stop.load();});

        m_sourceImage = image.clone();
        m_slamMapPoints = std::move(mapPoints);
        m_initialPose = pose.clone();
        m_runPrecompute = true;
        m_directTrackDataAvailable = true;
    }
}

void GLideEngine::initializeWindows()
{
    const int widthOffset = m_width + 80;
    const int heightOffset = m_height + 80;
    // m_windowFrames2D Tracking Window (Main OpenGL Context)
    m_windowFrames2D = GuiWindow::createWindow(50, heightOffset, m_width, m_height, m_windowFramesTitle);
    if (!m_windowFrames2D)
    {
       Logger::LogError("Viewer: Failed to initialize m_windowFrames2D tracking window.");
        return;
    }

    ensureWindowContext(m_windowFrames2D->getDisplay(),
                        m_windowFrames2D->getSurface(),
                        m_windowFrames2D->getContext());

    // shared OpenGL resources, valid for all
    //initializeBuffers();
    initializeShaders();

    m_trackLinesGfx = new Lines2D();
    m_trackLinesGfx->initializeEmptyBuffer();

    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    m_windowFrames2D->setEventCallback([this](const UIEvent& e){ this->onEvent(e); });

    // Mapping Window (shared)
    m_windowMap3D = new GuiWindow(widthOffset , 0, m_width, m_height,
                                    m_windowMapTitle,
                                    m_windowFrames2D->getContext(),
                                    m_windowFrames2D->getDisplay(),
                                    m_windowFrames2D->getConfig());

    ensureWindowContext(m_windowMap3D->getDisplay(),
                        m_windowMap3D->getSurface(),
                        m_windowMap3D->getContext());

    initializeMapPoints();

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    m_windowMap3D->setEventCallback([this](const UIEvent& e){ this->onEvent(e); });


    m_uiEventManager.subscribe(EventTypes::MouseMoved,[this](const UIEvent& e){ this->onMouse(e); });
    m_uiEventManager.subscribe(EventTypes::MousePressed,[this](const UIEvent& e){ this->onMouse(e); });
    m_uiEventManager.subscribe(EventTypes::MouseReleased,[this](const UIEvent& e){ this->onMouse(e); });
    m_uiEventManager.subscribe(EventTypes::MouseScrolled,[this](const UIEvent& e){ this->onMouse(e); });
    m_uiEventManager.subscribe(EventTypes::KeyPressed,[this](const UIEvent& e){ this->onKeyboard(e); });
    m_uiEventManager.subscribe(EventTypes::KeyReleased,[this](const UIEvent& e){ this->onKeyboard(e); });
    m_uiEventManager.subscribe(EventTypes::WindowClose,[this](const UIEvent& e){ this->onWindow(e); });
    m_uiEventManager.subscribe(EventTypes::WindowResize,[this](const UIEvent& e){ this->onWindow(e); });

    printVersions();
    Logger::LogInfoIII("Viewer: All windows initialized.");
}

void GLideEngine::render()
{
    //set context and do normal rendering
    ensureWindowContext(m_windowMap3D->getDisplay(), m_windowMap3D->getSurface(), m_windowMap3D->getContext());
    renderFrames2D();
    renderMap3D();
    PollEvents();
}

void GLideEngine::renderFrames2D()
{
    //set context and do normal rendering

    // if(checkUpdateFramesFlag())
    // {
    //     ensureWindowContext(m_windowFrames2D->getDisplay(), m_windowFrames2D->getSurface(), m_windowFrames2D->getContext());
    //     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //     m_canvasIndirectTracking->updateImage(m_canvasImage);
    //     m_trackLinesGfx->updatePoints(m_matchedFeature2DLines);
    //
    //     //render background images
    //     auto &canvasShader = m_shaders.find("canvasShader")->second;
    //     canvasShader->use();
    //     canvasShader->setUniform("TexSampler", 0);
    //     m_canvasIndirectTracking->render();
    //     glUseProgram(0);
    //
    //     //render tracking elements
    //     if (m_trackLinesGfx->getN() > 1)
    //     {
    //         auto &linesShader = m_shaders.find("linesShader")->second;
    //         linesShader->use();
    //         linesShader->setUniform("vRGB", m_featureLinesColor);
    //         m_trackLinesGfx->render();
    //         glUseProgram(0);
    //     }
    //
    //     clearUpdateFramesFlag();
    //
    //     m_windowFrames2D->onUpdateWindow();
    // }
}

void GLideEngine::renderMap3D()
{


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    m_mMatrix = glm::mat4(1.0f);
    setMatrices();

    auto &basicShader = m_shaders.find("basicShader")->second;
    basicShader->use();
    for (std::map<uint32_t, FrameGizmo* >::iterator it = m_keyFramesGfx.begin(); it != m_keyFramesGfx.end()
         ; it++)
    {
        m_mMatrix = it->second->getPose();
        setMatrices();
        basicShader->setUniform("vRGB", m_AllKeyFrameColor);
        basicShader->setUniform("mvpMatrix", m_mvpMatrix);
        it->second->render();
    }

    for (std::map<uint32_t, FrameGizmo* >::iterator it = m_tweenFramesDirectGfxCPU.begin(); it != m_tweenFramesDirectGfxCPU.end()
         ; it++)
    {
        m_mMatrix = it->second->getPose();
        setMatrices();
        basicShader->setUniform("vRGB", m_tweenFrameDirectColor);
        basicShader->setUniform("mvpMatrix", m_mvpMatrix);
        it->second->render();
    }

    for (std::map<uint32_t, FrameGizmo* >::iterator it = m_tweenFramesDirectGfxGPU.begin(); it != m_tweenFramesDirectGfxGPU.end()
         ; it++)
    {
        m_mMatrix = it->second->getPose();
        setMatrices();
        basicShader->setUniform("vRGB", glm::vec3(0.0,0.0,0.0));
        basicShader->setUniform("mvpMatrix", m_mvpMatrix);
        it->second->render();
    }



    for (std::map<uint32_t, FrameGizmo* >::iterator it = m_tweenFramesGfx.begin(); it != m_tweenFramesGfx.end()
         ; it++)
    {
        m_mMatrix = it->second->getPose();
        setMatrices();
        basicShader->setUniform("vRGB", m_tweenFrameColor);
        basicShader->setUniform("mvpMatrix", m_mvpMatrix);
        it->second->render();
    }

    //render current KF (main frame that shows up in 3D and active camera follows)
    m_mMatrix = m_currentKeyFrameGfx->getPose();
    float s = 2.0f;
    m_mMatrix = m_mMatrix * glm::scale(glm::mat4(1.0f), glm::vec3(s));
    setMatrices();
    basicShader->setUniform("vRGB", m_currentKeyFrameColor);
    basicShader->setUniform("mvpMatrix", m_mvpMatrix);
    m_currentKeyFrameGfx->render();
    glUseProgram(0);

    auto &pointShader = m_shaders.find("pointShader")->second;
    pointShader->use();
    //ref map points
    if(m_mapPointsRefGfx->getN()>0)
    {
        m_mMatrix = glm::mat4(1.0f);
        m_mMatrix[3].w = 1.0f;
        setMatrices();
        pointShader->setUniform("vRGB", m_mapPointsRefColor);
        pointShader->setUniform("pointSize", 3.0f);
        pointShader->setUniform("mvpMatrix", m_mvpMatrix);
        m_mapPointsRefGfx->render();
    }
    //all map points
    if(m_mapPointsGfx->getN()>0)
    {
        m_mMatrix = glm::mat4(1.0f);
        m_mMatrix[3].w = 1.0f;
        setMatrices();
        pointShader->setUniform("vRGB", m_mapPointsColor);
        pointShader->setUniform("pointSize", 2.0f);
        pointShader->setUniform("mvpMatrix", m_mvpMatrix);
        m_mapPointsGfx->render();
    }



    glUseProgram(0);

    glEnable(GL_DEPTH_TEST);
    m_windowMap3D->onUpdateWindow();
}

void GLideEngine::updateMapPoints()
{
    //make a local copy and load to buffer fetch pts addresses from map
    if (!m_stop)
    {

        std::vector<ORB_SLAM2::MapPoint *> mapPoints = m_map->GetAllMapPoints();
        std::vector<ORB_SLAM2::MapPoint *> mapRefPoints = m_map->GetReferenceMapPoints();

        std::vector<glm::vec3> mpVec3;
        uint32_t N = mapPoints.size();
        mpVec3.reserve(N);

        std::vector<glm::vec3> mpRefVec3;
        uint32_t NRef = mapRefPoints.size();
        mpRefVec3.reserve(NRef);

        for (auto &mp: mapPoints)
        {
            if (mp == nullptr)
                continue;

            const auto& pos = mp->GetWorldPos();

            //TODO: Convert directly to vec<GLFloat> here
            mpVec3.emplace_back(pos.at<float>(0)*m_scaleFactor,
                                -pos.at<float>(1)*m_scaleFactor,
                                pos.at<float>(2)*m_scaleFactor);
        }
        for (auto &mp: mapRefPoints)
        {
            if (mp == nullptr)
                continue;

            const auto& pos = mp->GetWorldPos();

            //TODO: Convert directly to vec<GLFloat> here
            mpRefVec3.emplace_back(pos.at<float>(0)*m_scaleFactor,
                                -pos.at<float>(1)*m_scaleFactor,
                                pos.at<float>(2)*m_scaleFactor);
        }

        //TODO: have function take vec<GLFLoat> directly
        m_mapPointsGfx->updatePoints(mpVec3);
        m_mapPointsRefGfx->updatePoints(mpRefVec3);

    }

}

void GLideEngine::updateFrames3D()
{
    if (!m_stop)
    {
        updateTweenIndirectFrames();
        updateTweenDirectFrames();
        updateKFrames();
    }
}

void GLideEngine::updateKFrames()
{
    //TODO: fix connection between frames (probably uses parent?)
    const std::vector<ORB_SLAM2::KeyFrame*> frames = m_map->GetAllKeyFrames();

    uint32_t lastKeyframeID = std::numeric_limits<uint32_t>::min();
    glm::mat4 lastKeyframePose = glm::mat4(1.0f);

    //use to convert: computer vision to computer graphics!
    glm::mat4 F(1.0f);
    F[1][1] = -1.0f;

    for (uint32_t n = 0; n < frames.size(); n++)
    {
        cv::Mat framePose = frames[n]->GetPoseInverse();

        glm::mat4 cvPose(1.0f);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                cvPose[j][i] = framePose.at<float>(i, j);

        glm::mat4 pose = F * cvPose * F;
        //scale
        pose[3].x *= m_scaleFactor;
        pose[3].y *= m_scaleFactor;
        pose[3].z *= m_scaleFactor;


        uint32_t id = frames[n]->mnFrameId;

        //update latest kf stuff
        if (id > lastKeyframeID)
        {
            lastKeyframeID = id;
            lastKeyframePose = pose;
            if (m_activeCamera->isFollowing())
                m_activeCamera->setTarget(lastKeyframePose);
            m_currentKeyFrameGfx->setPose(pose);
        }
        //if frame exists already, update pose
        if (m_keyFramesGfx.count(id))
        {
            m_keyFramesGfx[id]->setPose(pose);
        }
        //otherwise create new
        else
        {
            FrameGizmo* tempFrame = new FrameGizmo(0, pose, id);
            tempFrame->initialize();

            //if first frame (empty), there should be no parent
            if (!m_keyFramesGfx.empty())
            {
                tempFrame->setParentNode(std::prev(m_keyFramesGfx.end())->second);
            }
            m_keyFramesGfx[frames[n]->mnFrameId] = tempFrame;
        }
    }

    //Keyframes might be culled/deleted
    std::set<uint32_t> activeKeyFrames;
    for (size_t i = 0; i < frames.size(); i++)
        activeKeyFrames.insert(frames[i]->mnFrameId);

    //remove from frame gizmos frames that have been culled/removed
    for (std::map<uint32_t, FrameGizmo* >::iterator it = m_keyFramesGfx.begin(); it!=m_keyFramesGfx.end(); )
    {
        if (!activeKeyFrames.count(it->first))
        {
            delete it->second;
            it = m_keyFramesGfx.erase(it);
        }
        else it++;
    }

}

void GLideEngine::updateTweenIndirectFrames()
{
    const std::vector<ORB_SLAM2::Frame>& frames = m_map->GetTweenFrames();
    glm::mat4 F(1.0f);
    F[1][1] = -1.0f;
    //F[2][2] = -1.0f;

    for (uint32_t n = 0; n < frames.size(); n++)
    {
        cv::Mat framePose = frames[n].mTwc;

        glm::mat4 cvPose(1.0f);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                cvPose[j][i] = framePose.at<float>(i, j);

        glm::mat4 pose = F * cvPose * F;
        //scale
        pose[3].x *= m_scaleFactor;
        pose[3].y *= m_scaleFactor;
        pose[3].z *= m_scaleFactor;

        uint32_t id = frames[n].mnId;

        //if frame exists already, update pose
        if (m_tweenFramesGfx.count(id))
        {
            m_tweenFramesGfx[id]->setPose(pose);
        }
        //otherwise create new
        else
        {
            FrameGizmo* tempFrame = new FrameGizmo(0, pose, id);
            tempFrame->initialize();

            //if first frame (empty), there should be no parent
            if (!m_tweenFramesGfx.empty())
            {
                tempFrame->setParentNode(std::prev(m_tweenFramesGfx.end())->second);
            }
            m_tweenFramesGfx[frames[n].mnId] = tempFrame;
        }
    }
}

void GLideEngine::updateTweenDirectFrames()
{
    const std::vector<ORB_SLAM2::FrameDirect>& framesCPU = m_map->GetDirectTweenFramesCPU();
    const std::vector<ORB_SLAM2::FrameDirect>& framesGPU = m_map->GetDirectTweenFramesGPU();
    glm::mat4 F(1.0f);
    F[1][1] = -1.0f;
    //F[2][2] = -1.0f;

    for (uint32_t n = 0; n < framesCPU.size(); n++)
    {
        cv::Mat framePose = framesCPU[n].mTwc;

        glm::mat4 cvPose(1.0f);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                cvPose[j][i] = framePose.at<float>(i, j);

        glm::mat4 pose = F * cvPose * F;
        //scale
        pose[3].x *= m_scaleFactor;
        pose[3].y *= m_scaleFactor;
        pose[3].z *= m_scaleFactor;



        uint32_t id = framesCPU[n].mnId;

        //if frame exists already, update pose
        if (m_tweenFramesDirectGfxCPU.count(id))
        {
            m_tweenFramesDirectGfxCPU[id]->setPose(pose);
        }
        //otherwise create new
        else
        {
            FrameGizmo* tempFrame = new FrameGizmo(0, pose, id);
            tempFrame->initialize();

            //if first frame (empty), there should be no parent
            if (!m_tweenFramesDirectGfxCPU.empty())
            {
                tempFrame->setParentNode(std::prev(m_tweenFramesDirectGfxCPU.end())->second);
            }
            m_tweenFramesDirectGfxCPU[framesCPU[n].mnId] = tempFrame;
        }
    }

    for (uint32_t n = 0; n < framesGPU.size(); n++)
    {
        cv::Mat framePose = framesGPU[n].mTwc;

        glm::mat4 cvPose(1.0f);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                cvPose[j][i] = framePose.at<float>(i, j);

        glm::mat4 pose = F * cvPose * F;
        //scale
        pose[3].x *= m_scaleFactor;
        pose[3].y *= m_scaleFactor;
        pose[3].z *= m_scaleFactor;



        uint32_t id = framesGPU[n].mnId;

        //if frame exists already, update pose
        if (m_tweenFramesDirectGfxGPU.count(id))
        {
            m_tweenFramesDirectGfxGPU[id]->setPose(pose);
        }
        //otherwise create new
        else
        {
            FrameGizmo* tempFrame = new FrameGizmo(0, pose, id);
            tempFrame->initialize();

            //if first frame (empty), there should be no parent
            if (!m_tweenFramesDirectGfxGPU.empty())
            {
                tempFrame->setParentNode(std::prev(m_tweenFramesDirectGfxGPU.end())->second);
            }
            m_tweenFramesDirectGfxGPU[framesGPU[n].mnId] = tempFrame;
        }
    }
}

using namespace UIEvents;
void GLideEngine::PollEvents()
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        switch (event.type)
        {
            case SDL_WINDOWEVENT:
            {
                if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                {
                    int width = event.window.data1;
                    int height = event.window.data2;
                    WindowResizeUIEvent resizeEvent(width, height);
                    onEvent(resizeEvent);
                }
                else if (event.window.event == SDL_WINDOWEVENT_CLOSE)
                {
                    WindowCloseUIEvent closeEvent;
                    onEvent(closeEvent);
                }
                break;
            }

            case SDL_MOUSEBUTTONDOWN:
            {
                MouseButtonPressedUIEvent mouseEvent(event.button.button, true);
                onEvent(mouseEvent);
                break;
            }

            case SDL_MOUSEBUTTONUP:
            {
                MouseButtonReleasedUIEvent mouseEvent(event.button.button, false);
                onEvent(mouseEvent);
                break;
            }

            case SDL_MOUSEWHEEL:
            {
                MouseWheelUIEvent mouseEvent(event.wheel.y);
                onEvent(mouseEvent);
                break;
            }

            case SDL_MOUSEMOTION:
            {
                MouseMovedUIEvent mouseMoveEvent(
                    static_cast<float>(event.motion.x),
                    static_cast<float>(event.motion.y)
                );
                onEvent(mouseMoveEvent);
                break;
            }

            case SDL_KEYDOWN:
            {
                KeyPressUIEvent keyPressEvent(event.key.keysym.sym, 0);
                onEvent(keyPressEvent);
                break;
            }

            case SDL_KEYUP:
            {
                KeyReleaseUIEvent keyReleaseEvent(event.key.keysym.sym, 2);
                onEvent(keyReleaseEvent);
                break;
            }

            case SDL_QUIT:
            {
                stop();
                break;
            }
        }
    }
}


void GLideEngine::updateIndirectFeatureMatches(const cv::Mat &image, const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2, const std::vector<float> &d)
{
    if (!m_stop)
    {
        m_matchedFeature2DLines.clear();
        m_matchedFeature2DLines.reserve(2 * kpts1.size());

        // Precompute scaling factors for conversion to clip space
        static const float scaleX = 2.0f / (float) m_width;
        static const float scaleY = 2.0f / (float) m_height;

        for (uint32_t i = 0; i < kpts1.size(); i++)
        {
            m_matchedFeature2DLines.emplace_back((kpts1[i].pt.x * scaleX - 1.0f), (1.0f - kpts1[i].pt.y * scaleY), 0.0f);
            m_matchedFeature2DLines.emplace_back((kpts2[i].pt.x * scaleX - 1.0f), (1.0f - kpts2[i].pt.y * scaleY), 0.0f);
        }
    }
}

void GLideEngine::printVersions()
{

    const GLubyte *renderer = glGetString(GL_RENDERER);
    const GLubyte *vendor = glGetString(GL_VENDOR);
    const GLubyte *version = glGetString(GL_VERSION);
    const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

    GLint major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);


    printf("GL Vendor              : %s\n", vendor);
    printf("GL Renderer            : %s\n", renderer);
    printf("GL Version (string)    : %s\n", version);
    printf("GL Version (integeger) : %d.%d\n", major, minor);
    printf("GLSL Version           : %s\n", glslVersion);


    //query for supported extensions of the current OpenGL implementation
    bool logExtensions = false;
    if (logExtensions)

    {
        GLint nExtensions;
        glGetIntegerv(GL_NUM_EXTENSIONS, &nExtensions);

        for (int i = 0; i < nExtensions; i++)
            printf("%s\n", glGetStringi(GL_EXTENSIONS, i));
    }
}

void GLideEngine::shutdown()
{
    stop();

    //TODO: Make sure delete all allocated objects, deference pointers
    Logger::LogInfoI("Viewer: Shutting down.");

    // Delete map contents
    for (auto& pair : m_keyFramesGfx)
        delete pair.second;
    m_keyFramesGfx.clear();

    for (auto& pair : m_tweenFramesDirectGfxCPU)
        delete pair.second;
    m_tweenFramesDirectGfxCPU.clear();

    for (auto& pair : m_tweenFramesDirectGfxGPU)
        delete pair.second;
    m_tweenFramesDirectGfxGPU.clear();

    for (auto& pair : m_tweenFramesGfx)
        delete pair.second;
    m_tweenFramesGfx.clear();

    delete m_trackLinesGfx;
    delete m_canvasIndirectTracking;
    delete m_canvasDirectTracking;
    delete m_mapPointsGfx;
    delete m_mapPointsRefGfx;

    delete m_gpuCompute;


    // Exit windows BEFORE deleting
    if (m_windowFrames2D) {
        m_windowFrames2D->exit();
        delete m_windowFrames2D;
        m_windowFrames2D = nullptr;
    }

    if (m_windowMap3D) {
        m_windowMap3D->exit();
        delete m_windowMap3D;
        m_windowMap3D = nullptr;
    }

}

void GLideEngine::onMouse(const UIEvent &e)
{
    auto& eventType = e.getType();

    switch (eventType)
    {
        case EventTypes::MouseMoved:
        {
            const MouseMovedUIEvent* event = dynamic_cast<const MouseMovedUIEvent*>(&e);
            m_activeCamera->onLook(event->getX(),event->getY());
            break;
        }
        case EventTypes::MousePressed:
        {
            const MouseButtonPressedUIEvent* event = dynamic_cast<const MouseButtonPressedUIEvent*>(&e);
            //m_activeCamera->setIsFree(true);
            break;
        }
        case EventTypes::MouseReleased:
        {
            const MouseButtonReleasedUIEvent* event = dynamic_cast<const MouseButtonReleasedUIEvent*>(&e);
            //m_activeCamera->setIsFree(false);
            break;
        }
        case EventTypes::MouseScrolled:
        {
            const MouseWheelUIEvent* event = dynamic_cast<const MouseWheelUIEvent*>(&e);
            m_activeCamera->onZoom(event->getWheel());
            break;
        }
    }
}

void GLideEngine::onKeyboard(const UIEvent &e)
{
    auto& eventType = e.getType();
    switch (eventType)
    {
        case EventTypes::KeyPressed:
        {
            const KeyPressUIEvent* event = dynamic_cast<const KeyPressUIEvent*>(&e);
            m_activeCamera->onMove(event->getKey(),0);
            break;
        }
        case EventTypes::KeyReleased:
        {
            const KeyReleaseUIEvent* event = dynamic_cast<const KeyReleaseUIEvent*>(&e);
            int keyPRessed = event->getKey();

                //spacebar
                if (keyPRessed == 32)
                {
                    if (m_pauseSimulation.load())
                    {
                        Logger::LogInfoI("Viewer: Un-Pausing simulation.");
                        m_pauseSimulation.store(false);
                        //m_slamManager->onUnPause();
                    }
                    else
                    {
                        Logger::LogInfoI("Viewer: Pausing simulation.");
                        m_pauseSimulation.store(true);
                    }
                }
                //everything else
                else
                {
                    m_activeCamera->onMove(event->getKey(),2);
                }
            break;
        }
    }
}

void GLideEngine::onWindow(const UIEvent &e)
{
    auto& eventType = e.getType();
    switch (eventType)
    {
        case EventTypes::WindowClose:
        {
            //TODO: Implement logic to shutdown application
            break;
        }
        case EventTypes::WindowResize:
        {

            //TODO: Implement resize of windows
            break;
        }
    }

}

void GLideEngine::ensureWindowContext(EGLDisplay display, EGLSurface surface, EGLContext context)
{
    if (m_eglContext != context || m_eglSurface != surface || m_eglDisplay != display)
    {
        eglMakeCurrent(display, surface, surface,context);
        m_eglContext = context;
        m_eglSurface = surface;
        m_eglDisplay = display;
    }
}

bool GLideEngine::logTiming(const std::string& text)
{
    const std::string path = "gpuTimings.csv";

    if (!m_logTimingCreated)
    {
        std::ofstream init(path, std::ios::out | std::ios::trunc);
        if (!init.is_open())
        {
            Logger::LogError("Could not create/clear file: " + path);
            return false;
        }
        init << "function,frame,time_ms\n";
        init.close();
        m_logTimingCreated = true;
    }

    std::ofstream f(path, std::ios::out | std::ios::app);
    if (!f.is_open())
    {
        Logger::LogError("Could not open file: " + path);
        return false;
    }

    f << text << "\n";
    return true;
}

void GLideEngine::exit()
{
    shutdown();
}

void GLideEngine::stop()
{
    m_stop = true;
}

void GLideEngine::initializeShaders()
{
    //TODO: Remove all smart pointers -> Use raw pointers

    GLuint shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> shaderSimpleWhite = std::make_shared<Shader>();
    shaderSimpleWhite->setHandle(shaderProgram);
    shaderSimpleWhite->compile(GL_VERTEX_SHADER, "shaders/basicShader.vert");
    shaderSimpleWhite->compile(GL_FRAGMENT_SHADER, "shaders/basicShader.frag");
    shaderSimpleWhite->setShaderName("basicShader");
    shaderSimpleWhite->link();
    m_shaders["basicShader"] = shaderSimpleWhite;

    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> pointShader = std::make_shared<Shader>();
    pointShader->setHandle(shaderProgram);
    pointShader->compile(GL_VERTEX_SHADER, "shaders/pointShader.vert");
    pointShader->compile(GL_FRAGMENT_SHADER, "shaders/pointShader.frag");
    pointShader->setShaderName("pointShader");
    pointShader->link();
    m_shaders["pointShader"] = pointShader;

    //canvas shader
    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> shaderCanvas = std::make_shared<Shader>();
    shaderCanvas->setHandle(shaderProgram);
    shaderCanvas->compile(GL_VERTEX_SHADER, "shaders/canvasShader.vert");
    shaderCanvas->compile(GL_FRAGMENT_SHADER, "shaders/canvasShader.frag");
    shaderCanvas->setShaderName("canvasShader");
    shaderCanvas->link();
    m_shaders["canvasShader"] = shaderCanvas;

    //lines shader
    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> shaderLines = std::make_shared<Shader>();
    shaderLines->setHandle(shaderProgram);
    shaderLines->compile(GL_VERTEX_SHADER, "shaders/linesShader.vert");
    shaderLines->compile(GL_FRAGMENT_SHADER, "shaders/linesShader.frag");
    shaderLines->setShaderName("linesShader");
    shaderLines->link();
    m_shaders["linesShader"] = shaderLines;

    //compute shaders
    //image pyramid shaders, gauss resize
    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> convert8UCTo32FShader = std::make_shared<Shader>();
    convert8UCTo32FShader->setHandle(shaderProgram);
    convert8UCTo32FShader->compile(GL_COMPUTE_SHADER, "shaders/convert8UCTo32FShader.comp");
    convert8UCTo32FShader->setShaderName("convert8UCTo32FShader");
    convert8UCTo32FShader->link();
    m_shaders["convert8UCTo32FShader"] = convert8UCTo32FShader;

    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> gauss32FShader = std::make_shared<Shader>();
    gauss32FShader->setHandle(shaderProgram);
    gauss32FShader->compile(GL_COMPUTE_SHADER, "shaders/gauss32FShader.comp");
    gauss32FShader->setShaderName("gauss32FShader");
    gauss32FShader->link();
    m_shaders["gauss32FShader"] = gauss32FShader;

    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> resizeShader = std::make_shared<Shader>();
    resizeShader->setHandle(shaderProgram);
    resizeShader->compile(GL_COMPUTE_SHADER, "shaders/resizeShader.comp");
    resizeShader->setShaderName("resizeShader");
    resizeShader->link();
    m_shaders["resizeShader"] = resizeShader;

    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> preComputeShader = std::make_shared<Shader>();
    preComputeShader->setHandle(shaderProgram);
    preComputeShader->compile(GL_COMPUTE_SHADER, "shaders/preComputeShader.comp");
    preComputeShader->setShaderName("preComputeShader");
    preComputeShader->link();
    m_shaders["preComputeShader"] = preComputeShader;

    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> redPreComputeH1Shader = std::make_shared<Shader>();
    redPreComputeH1Shader->setHandle(shaderProgram);
    redPreComputeH1Shader->compile(GL_COMPUTE_SHADER, "shaders/redPreComputeH1Shader.comp");
    redPreComputeH1Shader->setShaderName("redPreComputeH1Shader");
    redPreComputeH1Shader->link();
    m_shaders["redPreComputeH1Shader"] = redPreComputeH1Shader;

    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> redPreComputeH2Shader = std::make_shared<Shader>();
    redPreComputeH2Shader->setHandle(shaderProgram);
    redPreComputeH2Shader->compile(GL_COMPUTE_SHADER, "shaders/redPreComputeH2Shader.comp");
    redPreComputeH2Shader->setShaderName("redPreComputeH2Shader");
    redPreComputeH2Shader->link();
    m_shaders["redPreComputeH2Shader"] = redPreComputeH2Shader;

    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> copyToSSBOShader = std::make_shared<Shader>();
    copyToSSBOShader->setHandle(shaderProgram);
    copyToSSBOShader->compile(GL_COMPUTE_SHADER, "shaders/copyToSSBOShader.comp");
    copyToSSBOShader->setShaderName("copyToSSBOShader");
    copyToSSBOShader->link();
    m_shaders["copyToSSBOShader"] = copyToSSBOShader;


    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> trackShader = std::make_shared<Shader>();
    trackShader->setHandle(shaderProgram);
    trackShader->compile(GL_COMPUTE_SHADER, "shaders/trackShader.comp");
    trackShader->setShaderName("trackShader");
    trackShader->link();
    m_shaders["trackShader"] = trackShader;

    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> redTrackShader = std::make_shared<Shader>();
    redTrackShader->setHandle(shaderProgram);
    redTrackShader->compile(GL_COMPUTE_SHADER, "shaders/redTrackShader.comp");
    redTrackShader->setShaderName("redTrackShader");
    redTrackShader->link();
    m_shaders["redTrackShader"] = redTrackShader;

    shaderProgram = glCreateProgram();
    std::shared_ptr<Shader> solveTrackShader = std::make_shared<Shader>();
    solveTrackShader->setHandle(shaderProgram);
    solveTrackShader->compile(GL_COMPUTE_SHADER, "shaders/solveTrackShader.comp");
    solveTrackShader->setShaderName("solveTrackShader");
    solveTrackShader->link();
    m_shaders["solveTrackShader"] = solveTrackShader;





}

void GLideEngine::initializeBuffers()
{
    glGenFramebuffers(1, &renderFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO);

    glGenRenderbuffers(1, &depthFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, depthFBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_width, m_height);

    GLenum drawBuffers[] = {GL_NONE};
    glDrawBuffers(1, drawBuffers);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    Logger::LogInfoIII("Viewer: Frame buffers initialized.");
}

void GLideEngine::initializeMapPoints()
{
    m_mapPointsGfx = new PointCloud();
    m_mapPointsGfx->initializeEmptyBuffer();

    m_mapPointsRefGfx = new PointCloud();
    m_mapPointsRefGfx->initializeEmptyBuffer();

    Logger::LogInfoIII("Viewer: Point cloud maps initialized.");
}

void GLideEngine::initializeCamera()
{
    glm::vec3 camPos(0.0f, 0.0f, -5.0f);
    glm::vec3 camTarget(0.0f, 0.0f, 1.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);

    m_activeCamera = std::make_shared<Camera>(m_width, m_height,camPos,camTarget,up);

    bool follow = m_GPUEngineSettings->gpuEngineParams.cameraFollow;
    const float followDistance = m_GPUEngineSettings->gpuEngineParams.followDistance;
    m_activeCamera->setFollow(follow, followDistance);
}

void GLideEngine::setMatrices()
{
    //because here we use OpenXR's matrices
    m_vMatrix = m_activeCamera->getViewMatrix();
    m_pMatrix = m_activeCamera->getProjectionMatrix();
    m_mvpMatrix = m_pMatrix * m_vMatrix * m_mMatrix;
}

void GLideEngine::setSquareUpdateFlag(const char &state)
{
    std::unique_lock<std::mutex> lock(m_stateMutex); {
        switch (state)
        {
            case 1: //startup
            {
                break;
            }
            case 2: //tracking
            {
                break;
            }
            case 3: //lost
            {
                break;
            }
            case 4: //recovery
            {
                break;
            }
        }
    }
}

bool GLideEngine::setActiveCamera(std::shared_ptr<Camera> camera)
{
    if (camera != nullptr)
    {
        m_activeCamera = camera;
        return true;
    }
    return false;
}

void GLideEngine::initializeProjectionMatrix()
{
    m_p = glm::perspective(glm::radians(m_fov),
                           static_cast<float>(m_width / m_height),
                           m_near,
                           m_far);

    std::cout << "Printing m_p matrix: " << std::endl;

    for (uint32_t i = 0; i < 4; i++)
        for (uint32_t j = 0; j < 4; j++)
            std::cout << std::to_string(m_p[j][i]) << std::endl;

    m_k = glm::mat4(1.0f);
    m_k[0][0] = 512;
    m_k[1][1] = 512;
    m_k[0][2] = 512;
    m_k[1][2] = 512;
}

void ViewerUtil::convertToGL(const std::vector<glm::vec3> &points, std::vector<GLfloat> &glPoints)
{
    glPoints.clear();
    for (uint32_t i = 0; i < points.size(); i++)
    {
        glPoints.push_back(points[i].x);
        glPoints.push_back(points[i].y);
        glPoints.push_back(points[i].z);
    }
}

void ViewerUtil::convertToGL(const std::vector<glm::vec2> &points, std::vector<GLfloat> &glPoints)
{
    glPoints.clear();
    for (uint32_t i = 0; i < points.size(); i++)
    {
        glPoints.push_back(points[i].x);
        glPoints.push_back(points[i].y);
    }
}

//********************************************************  SHADER ********************************************************

bool Shader::link()
{
    if (m_isLinked)
    {
        Logger::LogInfoI("Shader program has already been linked!");
        return false;
    }
    if (m_shaderProgram <= 0)
    {
        Logger::LogError("Link error: Shader program has not been created!");
        return false;
    }

    //atach compiled shaders
    for (size_t iLoop = 0; iLoop < m_compiledShaders.size(); iLoop++)
        glAttachShader(m_shaderProgram, m_compiledShaders[iLoop]);


    //link
    glLinkProgram(m_shaderProgram);

    GLint status;
    glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetProgramiv(m_shaderProgram, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetProgramInfoLog(m_shaderProgram, infoLogLength, NULL, strInfoLog);
        fprintf(stderr, "Linker failure: %s\n", strInfoLog);
        Logger::LogError(strInfoLog);
        delete[] strInfoLog;
        if (!m_shaderName.empty())
            Logger::LogError(m_shaderName + " shader linking failed!");
        return false;
    }

    //Link successful, get uniforms and set linked
    else
    {
        findUniformLocations();
        findAttributeLocations();
        m_isLinked = true;
    }

    //in either case, detach shader objects
    detachAndDeleteShaders();

    if (!m_shaderName.empty())
        Logger::LogInfoI(m_shaderName + " shader linked and loaded successfully.");
    return true;
}

std::string Shader::readFile(const std::string &path)
{
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f)
    {
        Logger::LogError("Error! File not found or could not be opened! " + path );
        return "";
    } else
    {
        Logger::LogError("Shader File found: " + path);
    }


    std::stringstream ss;
    ss << f.rdbuf();
    f.close();
    int length = 0;
    if (ss)
    {
        ss.seekg(0, ss.end);
        length = ss.tellg();
        ss.seekg(0, ss.beg);
        std::string shaderSource = ss.str();
        return shaderSource;
    } else
    {
        std::string error = "Error! File not found or could not be opened!" + path;
        return "";
    }
}

bool Shader::compile(GLenum shaderType, const std::string &shaderSrcFile)
{
    std::string shaderSource;
    GLuint shader = glCreateShader(shaderType);
    shaderSource = readFile(shaderSrcFile);
    const char *strFileData = shaderSource.c_str();

    if ((unsigned int) m_shaderProgram <= 0)
    {
        m_shaderProgram = glCreateProgram();
        if (m_shaderProgram == 0)
        {
            Logger::LogError("Unable to create shader program.");
        }
        return false;
    }

    glShaderSource(shader, 1, &strFileData, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetShaderInfoLog(shader, infoLogLength, NULL, strInfoLog);

        char *strShaderType;
        switch (shaderType)
        {
            case GL_VERTEX_SHADER: strShaderType = "vertex";
                break;
            case GL_GEOMETRY_SHADER: strShaderType = "geometry";
                break;
            case GL_FRAGMENT_SHADER: strShaderType = "fragment";
                break;
            case GL_COMPUTE_SHADER: strShaderType = "compute";
                break;
        }

        fprintf(stderr, "Compile failure in %s shader:\n%s\n", strShaderType, strInfoLog);
        delete[] strInfoLog;

        Logger::LogError("Compile failure in" + std::string(strShaderType) + "in shader: " + strInfoLog );

        return false;
    }

    m_compiledShaders.push_back(shader);
    return true;
}

void Shader::detachAndDeleteShaders()
{
    GLint numberOfShaders = 0;
    glGetProgramiv(m_shaderProgram, GL_ATTACHED_SHADERS, &numberOfShaders);
    std::vector<GLuint> shaderNames(numberOfShaders);
    glGetAttachedShaders(m_shaderProgram, numberOfShaders, NULL, shaderNames.data());
    for (GLuint attachedShader: shaderNames)
    {
        glDetachShader(m_shaderProgram, attachedShader);
        glDeleteShader(attachedShader);
    }
}

void Shader::findUniformLocations()
{
    m_uniformLocations.clear();

    GLint i;
    GLint count;
    GLint size; // size of the variable
    GLenum type; // type of the variable (float, vec3 or mat4, etc)

    const GLsizei bufSize = 64; // maximum name length
    GLchar name[bufSize]; // variable name in GLSL
    GLsizei length; // name length
    glGetProgramiv(m_shaderProgram, GL_ACTIVE_UNIFORMS, &count);
    for (i = 0; i < count; i++)
    {
        glGetActiveUniform(m_shaderProgram, (GLuint) i, bufSize, &length, &size, &type, name);

        printf("Uniform #%d Type: %u Name: %s\n", i, type, name);
        m_uniformLocations[name] = glGetUniformLocation(m_shaderProgram, name);
    };
}

void Shader::findAttributeLocations()
{
    GLint i;
    GLint count;
    GLint size; // size of the variable
    GLenum type; // type of the variable (float, vec3 or mat4, etc)

    const GLsizei bufSize = 16; // maximum name length
    GLchar name[bufSize]; // variable name in GLSL
    GLsizei length; // name length
    glGetProgramiv(m_shaderProgram, GL_ACTIVE_ATTRIBUTES, &count);
    printf("Active Attributes: %d\n", count);

    for (i = 0; i < count; i++)
    {
        glGetActiveAttrib(m_shaderProgram, (GLuint) i, bufSize, &length, &size, &type, name);

        printf("Attribute #%d Type: %u Name: %s\n", i, type, name);
        m_attributeLocations[name] = glGetAttribLocation(m_shaderProgram, name);;
    }
}

void Shader::setUniform(const char *name, const glm::mat4 &m)
{
    GLint loc = getUniformLocation(name);
    glUniformMatrix4fv(loc, 1, GL_FALSE, &m[0][0]);
}

void Shader::setUniform(const char *name, const glm::vec3 &v)
{
    GLint loc = getUniformLocation(name);
    glUniform3f(loc, v.x, v.y, v.z);
}

void Shader::setUniform(const char *name, const glm::vec2 &v)
{
    GLint loc = getUniformLocation(name);
    glUniform2f(loc, v.x, v.y);
}

void Shader::setUniform(const char *name, float val)
{
    GLint loc = getUniformLocation(name);
    glUniform1f(loc, val);
}

void Shader::setUniform(const char *name, int val)
{
    GLint loc = getUniformLocation(name);
    glUniform1i(loc, val);
}

int Shader::getUniformLocation(const char *name)
{
    //traverse map
    auto index = m_uniformLocations.find(name);

    //not found, means uniform was not included
    if (index == m_uniformLocations.end())
    {
        //case uniform is not present in map, include in map if uniform location is valid ( > 0)
        GLint loc = glGetUniformLocation(m_shaderProgram, name);
        if (loc >= 0)
        {
            m_uniformLocations[name] = loc;
            return loc;
        } else
        {
            std::string output = name;
            output = "uniform: " + output + " not found!";
            fprintf(stderr, "%s", output.c_str());
            return -1;
        }
    }
    return index->second;
}

bool Shader::setHandle(GLuint handle)
{
    if (handle != 0)
    {
        m_shaderProgram = handle;
        return true;
    } else
        std::cerr << "invalid shader program!" << std::endl;
    return false;
}

//********************************************************  CAMERA ********************************************************

void Camera::update(float t)
{
    if (!m_follow)
    {
        updateMove(t);
        setTransform();
    }
    else
    {
        follow();
    }
}

void Camera::updateMove(float t)
{
    const float maxSpeed = 20.0f;
    const float speedStep = 0.25f;

    //float something = smoothStep(somevariable+=speedStep, 0, 1)*maxSpeed;

    //forward backward
    if (m_keyMap & 1)
    {
        if(m_motion.vForward < 0)
        {
            m_motion.vForward += m_motion.deacceleration;
        }
        m_motion.vForward += ViewerUtil::smoothStep(m_motion.iForward += speedStep, 0, maxSpeed);
        m_motion.vForward = (m_motion.vForward > maxSpeed) ? maxSpeed : m_motion.vForward;
    }
    else if (m_keyMap & 2)
    {
        if(m_motion.vForward > 0)
        {
            m_motion.vForward -= m_motion.deacceleration;
        }
        m_motion.vForward -= ViewerUtil::smoothStep(m_motion.iForward += speedStep, 0, maxSpeed);
        m_motion.vForward = (m_motion.vForward < -maxSpeed) ? -maxSpeed : m_motion.vForward;
    }
    else
    {
        m_motion.iForward = 0;

        if (abs(m_motion.vForward) > m_motion.stopSpeed)
        {
            m_motion.vForward *= m_motion.deacceleration;
        }
        else
            m_motion.vForward = 0.0f;

    }

    if (m_keyMap & 4)
    {
        if(m_motion.vSide < 0)
        {
            m_motion.vSide += m_motion.deacceleration;
        }
        m_motion.vSide += ViewerUtil::smoothStep(m_motion.iSide += speedStep, 0, maxSpeed);
        m_motion.vSide = (m_motion.vSide > maxSpeed) ? maxSpeed : m_motion.vSide;
    }
    else if (m_keyMap & 8)
    {
        if(m_motion.vSide > 0)
        {
            m_motion.vSide -= m_motion.deacceleration;
        }
        m_motion.vSide -= ViewerUtil::smoothStep(m_motion.iSide += speedStep, 0, maxSpeed);
        m_motion.vSide = (m_motion.vSide < -maxSpeed) ? -maxSpeed : m_motion.vSide;
    }
    else
    {
        m_motion.iSide = 0;
        if (abs(m_motion.vSide) > m_motion.stopSpeed)
        {
            m_motion.vSide *= m_motion.deacceleration;
        }
        else
            m_motion.vSide = 0.0f;
    }


    if(m_zoom > 0)
    {
        if(m_motion.vForward < 0)
        {
            m_motion.vForward += m_motion.deacceleration;
        }
        m_motion.vForward += m_motion.iZoom;
        m_motion.vForward = (m_motion.vForward > maxSpeed) ? maxSpeed : m_motion.vForward;
    }
    else if(m_zoom < 0)
    {
        if(m_motion.vForward > 0)
        {
            m_motion.vForward -= m_motion.deacceleration;
        }
        m_motion.vForward -= m_motion.iZoom;
        m_motion.vForward = (m_motion.vForward < -maxSpeed) ? -maxSpeed : m_motion.vForward;
    }
    m_zoom = 0;

    m_position = m_position + (m_forward * m_stepSensitivity * m_motion.vForward * t);
    m_position = m_position + (m_right * m_stepSensitivity * m_motion.vSide * t);
}

void Camera::setTransform()
{
    glm::mat4 posMat(1.0f);
    posMat[3] = glm::vec4(m_position,1.0f);

    glm::mat4 rotMat(1.0);
    rotMat[0] = glm::vec4(m_right, 0.0f);
    rotMat[1] = glm::vec4(m_up, 0.0f);
    rotMat[2] = glm::vec4(-m_forward, 0.0f);
    m_orientation[0] = glm::vec3(rotMat[0]);
    m_orientation[1] = glm::vec3(rotMat[1]);
    m_orientation[2] = glm::vec3(rotMat[2]);

    //TODO: rename this to camW and camC or something mor coherent
    //world space
    glm::mat4 m_transform = posMat * rotMat;

    //camera space
    m_viewMatrix = glm::transpose(m_orientation); // Inverse rotation matrix
    m_viewMatrix[3] = glm::vec4(-glm::transpose(m_orientation) * m_position, 1.0f);
}

void Camera::setProjectionTransform()
{
    m_projectionMatrix = glm::perspective(glm::radians(m_fov),static_cast<float>(m_width/m_height),m_near,m_far);
}

void Camera::initialize()
{
    m_projectionMatrix = glm::mat4(1.0f);
    m_viewMatrix = glm::mat4(1.0f);
    m_viewProjectionMatrix= glm::mat4(1.0f);
    m_forward = glm::normalize(m_forward);
    m_up = glm::normalize(m_up);
    m_right = glm::cross(m_up, m_forward);
    m_right = glm::normalize(m_right);
    m_fov = 90;
    m_near = 1;
    m_far = 1000;

    setProjectionTransform();

    m_horAngle = -90.0f;
    m_verAngle = 0;
}

void Camera::onMove(int key, int mode)
{
    switch (key)
    {
        case 119: //w, front
        {
            m_keyMap = (mode == 2) ? m_keyMap & ~(0x01) : m_keyMap = m_keyMap | 1;
            break;
        }
        case 115: //s, back
        {
            m_keyMap = (mode == 2) ? m_keyMap & ~(0x02) : m_keyMap = m_keyMap | 2;
            break;
        }
        case 100: //d, right
        {
            m_keyMap = (mode == 2) ? m_keyMap & ~(0x04) : m_keyMap = m_keyMap | 4;
            break;
        }
        case 97: //a, left
        {
            m_keyMap = (mode == 2) ? m_keyMap & ~(0x08) : m_keyMap = m_keyMap | 8;
            break;
        }
        default:
        {
            break;
        }
    }

}

void Camera::follow()
{
    // Extract orientation (3x3 rotation)
    glm::mat3 R = glm::mat3(m_target);

    // Extract keyframe position
    glm::vec3 kfPos = glm::vec3(m_target[3]);

    // Forward is the Z column (third column of rotation)
    glm::vec3 forward = glm::normalize(glm::vec3(R[2]));

    // Position camera behind keyframe
    m_position = kfPos - forward * m_followDistance;

    // Copy orientation vectors
    m_right = glm::normalize(glm::vec3(R[0]));
    m_up = glm::normalize(glm::vec3(R[1]));
    m_forward = forward;

    setTransform();
}

void Camera::onLook(int x, int y)
{
    if (!m_follow)
    {
        //temp vectors
        glm::vec3 right(1.0f, 0.0f, 0.0f);
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        glm::vec3 target(1.0f, 0.0f, 0.0f);


        //update mouse to current cursor pos
        if (!m_isInitialized)
        {
            m_isInitialized = true;
            m_mouseX = (float)x;
            m_mouseY = (float)y;
        }


        m_dx = ((float)x - m_mouseX) * m_horSensitivity;
        m_dy = ((float)y - m_mouseY) * m_verSensitivity;

        m_mouseX = (float)x;
        m_mouseY = (float)y;

        const float alpha = 0.85f;

        m_sdx = (1.0f-alpha)*m_sdx + static_cast<float>(m_dx)*alpha;
        m_sdy = (1.0f-alpha)*m_sdy + static_cast<float>(m_dy)*alpha;

        m_horAngle += m_sdx / 3.0f;
        m_verAngle += m_sdy / 3.0f;

        m_verAngle = (m_verAngle < m_maxVerAngle) ? m_maxVerAngle : m_verAngle;
        m_verAngle = (m_verAngle > -m_maxVerAngle) ? -m_maxVerAngle : m_verAngle;

        float horizontalAngleRad = glm::radians(m_horAngle);
        float verticalAngleRad = glm::radians(m_verAngle);

        glm::vec3 verticalVector(0.0f, 1.0f, 0.0f);
        glm::vec3 viewVector(1.0f, 0.0f, 0.0f);

        viewVector = ViewerUtil::rotateAngleAxis(viewVector, horizontalAngleRad, verticalVector);
        viewVector = glm::normalize(viewVector);

        glm::vec3 rightVector = glm::cross(verticalVector, viewVector);
        rightVector = glm::normalize(rightVector);

        viewVector = ViewerUtil::rotateAngleAxis(viewVector, verticalAngleRad, rightVector);
        viewVector = glm::normalize(viewVector);

        glm::vec3 upVector = glm::cross(viewVector, rightVector);
        upVector = glm::normalize(upVector);

        m_forward = viewVector;
        m_up = upVector;
        m_right = rightVector;

        setTransform();
    }
    else
    {
        m_isInitialized = false;
    }
}

//********************************************************  CANVAS ********************************************************

void Canvas::updateImage(const cv::Mat &image)
{
    m_image = image.clone();
    cv::cvtColor(m_image, m_image, cv::COLOR_BGR2RGB);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, m_image.data);
}

void Canvas::render() const
{
    if (m_vao != 0)
    {
        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(m_vao);


#if GLIDE_DEBUG_GL_ERRORS
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) std::cout << "VAO bind error: " << err << std::endl;
#endif
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_texture);


#if GLIDE_DEBUG_GL_ERRORS
        err = glGetError();
        if (err != GL_NO_ERROR) std::cout << "Texture bind error: " << err << std::endl;
#endif
        glDrawElements(GL_TRIANGLES, m_N, GL_UNSIGNED_INT, 0);

#if GLIDE_DEBUG_GL_ERRORS
        err = glGetError();
        if (err != GL_NO_ERROR) std::cout << "Draw error: " << err << std::endl;
#endif
        glBindVertexArray(0);
        glEnable(GL_DEPTH_TEST);
    }
}

void Canvas::initialize()
{
    std::vector<glm::vec3> points;
    std::vector<GLfloat> glpoints;
    std::vector<GLuint> indices;
    std::vector<GLfloat> glTexCoords;

    //quad vertices
    points.emplace_back(-1, -1, 0); //left lower
    points.emplace_back(-1, 1, 0); //left upper
    points.emplace_back(1, 1, 0); //right upper
    points.emplace_back(1, -1, 0); //right lower

    //indices
    indices = {0, 2, 1, 0, 3, 2};

    //texture coordinates
    std::vector<glm::vec2> texCoords;
    texCoords.emplace_back(0, 1);
    texCoords.emplace_back(0, 0);
    texCoords.emplace_back(1, 0);
    texCoords.emplace_back(1, 1);


    ViewerUtil::convertToGL(points, glpoints);
    ViewerUtil::convertToGL(texCoords, glTexCoords);

    initializeBuffers(&glpoints, &indices, &glTexCoords);
    bindTexture();
}

void Canvas::initializeBuffers(std::vector<GLfloat> *glPoints, std::vector<GLuint> *indices,std::vector<GLfloat> *texCoords)
{
    if (!m_buffers.empty()) deleteBuffers();

    // Must have data for points, indices, texture coordinates
    if (indices == nullptr || glPoints == nullptr || texCoords == nullptr)
    {
        return;
    }

    m_N = indices->size();

    GLuint posBuffer = 0, indexBuffer = 0, textCoordBuffer = 0;

    //index
    glGenBuffers(1, &indexBuffer);
    m_buffers.push_back(indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices->size() * sizeof(GLuint), indices->data(), GL_STATIC_DRAW);

    //points position
    glGenBuffers(1, &posBuffer);
    m_buffers.push_back(posBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
    glBufferData(GL_ARRAY_BUFFER, glPoints->size() * sizeof(GLfloat), glPoints->data(), GL_STATIC_DRAW);

    //texture coords.
    glGenBuffers(1, &textCoordBuffer);
    m_buffers.push_back(textCoordBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, textCoordBuffer);
    glBufferData(GL_ARRAY_BUFFER, texCoords->size() * sizeof(GLfloat), texCoords->data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    //index
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);

    // Position
    glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0); // Vertex position

    // Tex coords
    glBindBuffer(GL_ARRAY_BUFFER, textCoordBuffer);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1); // Tex coord

    glBindVertexArray(0);
}

void Canvas::bindTexture()
{
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0,GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void Canvas::deleteBuffers()
{
    if (m_buffers.size() > 0)
    {
        glDeleteBuffers((GLsizei) m_buffers.size(), m_buffers.data());
        m_buffers.clear();
    }

    if (m_vao != 0)
    {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
}

//********************************************************  GL ELEMENTS ********************************************************

void GraphicPrimitive::initialize()
{
}

void GraphicPrimitive::loadPoints(const std::vector<glm::vec3> &points, const std::vector<glm::vec3> &pointsColor)
{
    std::vector<GLfloat> glPoints;
    std::vector<GLfloat> glPointsColor;

    ViewerUtil::convertToGL(points, glPoints);
    ViewerUtil::convertToGL(pointsColor, glPointsColor);
    initializeBuffers(&glPoints, &glPointsColor);
}

void GraphicPrimitive::loadPoints(const std::vector<glm::vec3> &points)
{
    std::vector<GLfloat> glPoints;
    ViewerUtil::convertToGL(points, glPoints);
    initializeBuffers(&glPoints);
}

void GraphicPrimitive::initializeBuffers(std::vector<GLfloat> *glPoints, std::vector<GLfloat> *glpointsColors)
{
    if (glPoints == nullptr || glpointsColors == nullptr)
        return;

    GLuint posBuffer = 0, colorBuffer = 0;
    int vtxPosAttributeIndex = 0;
    int vtxColorAttributeIndex = 1;
    m_N = glPoints->size();


    glGenBuffers(1, &posBuffer);
    m_buffers.push_back(posBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
    glBufferData(GL_ARRAY_BUFFER, glPoints->size() * sizeof(GLfloat), glPoints->data(), GL_STATIC_DRAW);

    glGenBuffers(1, &colorBuffer);
    m_buffers.push_back(colorBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, glpointsColors->size() * sizeof(GLfloat), glpointsColors->data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
    glVertexAttribPointer(vtxPosAttributeIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(vtxPosAttributeIndex); // Vertex position

    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glVertexAttribPointer(vtxColorAttributeIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(vtxColorAttributeIndex);

    glBindVertexArray(0);
}

void GraphicPrimitive::initializeBuffers(std::vector<GLfloat> *glPoints)
{
    if (glPoints == nullptr)
        return;

    GLuint posBuffer = 0, colorBuffer = 0;
    m_N = glPoints->size() / 3;

    glGenBuffers(1, &posBuffer);
    m_buffers.push_back(posBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
    glBufferData(GL_ARRAY_BUFFER, glPoints->size() * sizeof(GLfloat), glPoints->data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0); // Vertex position

    glBindVertexArray(0);
}

void GraphicPrimitive::render() const
{
    if (m_vao != 0)
    {
        glBindVertexArray(m_vao);
        glDrawArrays(GL_LINES, 0, m_N);
        glBindVertexArray(0);
    }
}

void GraphicPrimitive::deleteBuffers()
{
    if (m_buffers.size() > 0)
    {
        glDeleteBuffers((GLsizei) m_buffers.size(), m_buffers.data());
        m_buffers.clear();
    }

    if (m_vao != 0)
    {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
}

void GraphicPrimitive::updateBuffer(const std::vector<GLfloat> *glPoints)
{
    if ((!glPoints) || (glPoints->size() <3))
        return;

    m_N = (int)(glPoints->size()/3);
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_buffers[0]);
    //orphan old storage (will clear and create a new one so doesnt stall cpu)
    glBufferData(GL_ARRAY_BUFFER, glPoints->size() * sizeof(GLfloat), nullptr, GL_STREAM_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, glPoints->size()*sizeof(GLfloat), glPoints->data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void GraphicPrimitive::updateBuffer(const std::vector<GLfloat> *glPoints,const std::vector<GLfloat> *glpointsColors)
{
    if ((!glPoints) || (!glpointsColors)) return;
    if ((glPoints->size() < 3) || (glpointsColors->size() < 3)) return;

    m_N = (int)(glPoints->size()/3);
    glBindBuffer(GL_ARRAY_BUFFER, m_buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, glPoints->size() * sizeof(GLfloat), nullptr, GL_STREAM_DRAW);
    glBufferData(GL_ARRAY_BUFFER, glPoints->size() * sizeof(GLfloat), glPoints->data(), GL_DYNAMIC_DRAW);


    glBindBuffer(GL_ARRAY_BUFFER, m_buffers[1]);
    glBufferData(GL_ARRAY_BUFFER, glpointsColors->size() * sizeof(GLfloat), nullptr, GL_STREAM_DRAW);
    glBufferData(GL_ARRAY_BUFFER, glpointsColors->size() * sizeof(GLfloat), glpointsColors->data(), GL_DYNAMIC_DRAW);


    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}


void GraphicPrimitive::initializeEmptyBuffer()
{
    // Generate VAO
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    // Generate VBO
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);


    // Initialize VBO with an empty buffer
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    // Set up vertex attributes (if needed)
    // For example, assuming points are 3D coordinates stored as floats:
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, nullptr);
    glEnableVertexAttribArray(0);

    // Unbind VAO and VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    m_buffers.push_back(m_vbo);
}

void GraphicPrimitive::initializeBuffers(const std::vector<GLfloat> *points, const std::vector<GLuint> *indices)
{
    if (!m_buffers.empty()) deleteBuffers();

    m_N = points->size() / 2;
    int vtxAttributeIndex = 0;


    // Must have data for indices, points
    if (indices == nullptr || points == nullptr)
    {
        return;
    }


    GLuint indexBuf = 0, posBuf = 0, normBuf = 0, tcBuf = 0, tangentBuf = 0, vertexColorsBuf = 0;
    glGenBuffers(1, &indexBuf);
    m_buffers.push_back(indexBuf);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuf);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices->size() * sizeof(GLuint), indices->data(), GL_STATIC_DRAW);

    glGenBuffers(1, &posBuf);
    m_buffers.push_back(posBuf);
    glBindBuffer(GL_ARRAY_BUFFER, posBuf);
    glBufferData(GL_ARRAY_BUFFER, points->size() * sizeof(GLfloat), points->data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuf);

    // Position
    glBindBuffer(GL_ARRAY_BUFFER, posBuf);
    glVertexAttribPointer(vtxAttributeIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(vtxAttributeIndex); // Vertex position
    vtxAttributeIndex++;

    glBindVertexArray(0);
}

void AxisGizmo::initialize()
{
    std::vector<glm::vec3> points;
    points.push_back(glm::vec3(0, 0, 0));
    points.push_back(glm::vec3(-1, 0, 0));

    points.push_back(glm::vec3(0, 0, 0));
    points.push_back(glm::vec3(0, 1, 0));

    points.push_back(glm::vec3(0, 0, 0));
    points.push_back(glm::vec3(0, 0, 1));

    std::vector<glm::vec3> pointsColors;
    pointsColors.push_back(glm::vec3(1, 0, 0));
    pointsColors.push_back(glm::vec3(1, 0, 0));

    pointsColors.push_back(glm::vec3(0, 1, 0));
    pointsColors.push_back(glm::vec3(0, 1, 0));

    pointsColors.push_back(glm::vec3(0, 0, 1));
    pointsColors.push_back(glm::vec3(0, 0, 1));
    loadPoints(points, pointsColors);
}

void FrameGizmo::initialize()
{
    //update position and rotation matrix
    m_R[0] = m_pose[0];
    m_R[1] = m_pose[1];
    m_R[2] = m_pose[2];
    m_t = m_pose[3];


    std::vector<glm::vec3> points;
    //top
    points.push_back(glm::vec3(-0.5, 0.5, 0.5));
    points.push_back(glm::vec3(0.5, 0.5, 0.5));

    //right
    points.push_back(glm::vec3(0.5, 0.5, 0.5));
    points.push_back(glm::vec3(0.5, -0.5, 0.5));

    //bottom
    points.push_back(glm::vec3(0.5, -0.5, 0.5));
    points.push_back(glm::vec3(-0.5, -0.5, 0.5));

    //left
    points.push_back(glm::vec3(-0.5, -0.5, 0.5));
    points.push_back(glm::vec3(-0.5, 0.5, 0.5));

    //center
    points.push_back(glm::vec3(0.0, 0.0, 0.0));
    points.push_back(glm::vec3(-0.5, 0.5, 0.5));

    points.push_back(glm::vec3(0.0, 0.0, 0.0));
    points.push_back(glm::vec3(0.5, 0.5, 0.5));

    points.push_back(glm::vec3(0.0, 0.0, 0.0));
    points.push_back(glm::vec3(0.5, -0.5, 0.5));

    points.push_back(glm::vec3(0.0, 0.0, 0.0));
    points.push_back(glm::vec3(-0.5, -0.5, 0.5));


    loadPoints(points);
}

void FrameGizmo::setParentNode(FrameGizmo *frameGizmo)
{
    m_parent = frameGizmo;
}

void FrameGizmo::renderAxisGizmos() const
{
    m_axisGizmo.render();
}

void PathGizmo::updatePoints(const std::vector<glm::vec3> &points)
{
    m_N = points.size();
    std::vector<GLfloat> glPoints;
    ViewerUtil::convertToGL(points, glPoints);
    updateBuffer(&glPoints);
}

void PathGizmo::render() const
{
    GraphicPrimitive::render();
}

void Lines2D::updatePoints(const std::vector<glm::vec3> &points)
{
    m_N = points.size();
    std::vector<GLfloat> glPoints;
    ViewerUtil::convertToGL(points, glPoints);
    updateBuffer(&glPoints);
}

void Lines2D::render() const
{
    GraphicPrimitive::render();
}

void TrailGizmo::render() const
{
    if (m_vao == 0) return;

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, m_N);
    glBindVertexArray(0);
}

void PointCloud::render() const
{
    if (m_vao == 0) return;

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, m_N);
    glBindVertexArray(0);
}

void PointCloud::initializeEmptyBuffer()
{
    // Generate VAO
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    // Generate VBO for positions
    GLuint posBuffer;
    glGenBuffers(1, &posBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW); // Empty buffer for positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr); // Attribute 0: positions
    glEnableVertexAttribArray(0);

    // Generate VBO for colors
    GLuint colorBuffer;
    glGenBuffers(1, &colorBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW); // Empty buffer for colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr); // Attribute 1: colors
    glEnableVertexAttribArray(1);

    // Unbind VAO and buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Store the buffer IDs
    m_buffers.push_back(posBuffer);
    m_buffers.push_back(colorBuffer);
}

//TODO: use single conversion for points and frames etc.
void PointCloud::updatePoints(const std::vector<glm::vec3> &points)
{
    std::vector<GLfloat> glPoints;
    ViewerUtil::convertToGL(points, glPoints);
    updateBuffer(&glPoints);
}

void PointCloud::updatePoints(const std::vector<glm::vec3> &points,const std::vector<glm::vec3> &pointsColor)
{
    std::vector<GLfloat> glPoints;
    std::vector<GLfloat> glpointsColors;

    ViewerUtil::convertToGL(points, glPoints);
    ViewerUtil::convertToGL(pointsColor, glpointsColors);
    updateBuffer(&glPoints,&glpointsColors);
}

GuiWindow::GuiWindow() : m_width(800), m_height(600), m_title("ImageSLAM")
{
    Logger::LogInfoIV("\n Viewer: Creating window:" + m_title);
    initializeWindow(EGL_NO_CONTEXT);
}

GuiWindow::GuiWindow(int x, int y, int width, int height, const std::string &title) : m_xOffset(x), m_yOffset(y), m_width(width), m_height(height), m_title(title)
{
    Logger::LogInfoIV("\n Viewer: Creating window:" + m_title);
    initializeWindow(EGL_NO_CONTEXT);
}

//Share resources between windows
GuiWindow::GuiWindow(int x, int y, int width, int height, const std::string &title, EGLContext otherContext, EGLDisplay otherDisplay, EGLConfig otherConfig)
    : m_xOffset(x), m_yOffset(y), m_width(width),m_height(height), m_title(title)
{
    Logger::LogInfoIV("\n Viewer: Creating window:" + m_title);
    initializeWindowShared(otherContext, otherDisplay, otherConfig);
}

GuiWindow::~GuiWindow()
{
}

GuiWindow *GuiWindow::createWindow()
{
    return new GuiWindow();
}

GuiWindow *GuiWindow::createWindow(int x, int y, int width, int height, const std::string &title)
{
    return new GuiWindow(x, y, width, height, title);
}

//Share resources
GuiWindow *GuiWindow::createWindow(int x, int y, int width, int height, const std::string &title, EGLContext otherContext, EGLDisplay otherDisplay, EGLConfig otherConfig)
{
    return new GuiWindow(x, y, width, height, title, otherContext, otherDisplay, otherConfig);
}

bool GuiWindow::initializeWindowShared(EGLContext sharedContext, EGLDisplay sharedDisplay, EGLConfig sharedConfig)
{
    const EGLint configAttribs[] =
    {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_NONE
    };

    EGLint contextAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION_KHR, 3,
        EGL_NONE
    };


    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        Logger::LogError("Unable to initialize SDL: " + std::string(SDL_GetError()));
        return false;
    } else
    {
        Logger::LogInfoI("Initialized SDL" + std::string(SDL_GetError()));
    }

    int displayIndex = 0;
    SDL_Rect displayBounds;
    if (SDL_GetDisplayBounds(displayIndex, &displayBounds) < 0)
    {
        Logger::LogError("Failed to get display bounds");
    }

    m_window = SDL_CreateWindow(m_title.c_str(), displayBounds.x + m_xOffset, displayBounds.y + m_yOffset,
                                m_width, m_height, SDL_WINDOW_OPENGL);
    if (m_window == NULL)
    {
        Logger::LogError("Unable to create window SDL: " + std::string(SDL_GetError()));
        SDL_Quit();
        return false;
    } else
    {
        Logger::LogInfoI("Created window SDL" + std::string(SDL_GetError()));
    }


    // ------------------------------------------------------------
    // Use the main window’s EGLDisplay and EGLConfig
    // ------------------------------------------------------------
    if (sharedDisplay != EGL_NO_DISPLAY)
        m_eglDisplay = sharedDisplay;
    else
        m_eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    if (sharedConfig)
        m_eglConfig = sharedConfig;
    else {
        EGLint numConfigs;
        if (!eglChooseConfig(m_eglDisplay, configAttribs, &m_eglConfig, 1, &numConfigs))
        {
            EGLint error = eglGetError();
            const char *errorMessage = eglGetErrorString(error);
            Logger::LogError("Failed to choose EGL config" + std::string(errorMessage));
            SDL_DestroyWindow(m_window);
            SDL_Quit();
            return false;
        }
    }


    SDL_SysWMinfo sysInfo;
    SDL_VERSION(&sysInfo.version);
    SDL_GetWindowWMInfo(m_window, &sysInfo);


    if (!eglBindAPI(EGL_OPENGL_ES_API))
    {
        EGLint error = eglGetError();
        Logger::LogError("eglBindAPI(OpenGL ES) failed: " + std::to_string(error));
        return false;
    }

    // ------------------------------------------------------------
    // Create a new context sharing with the main one
    // ------------------------------------------------------------
    m_eglContext = eglCreateContext(m_eglDisplay, m_eglConfig, sharedContext, contextAttribs);
    if (m_eglContext == EGL_NO_CONTEXT)
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogError("Failed to create shared EGL context" + std::string(errorMessage));
        SDL_DestroyWindow(m_window);
        SDL_Quit();
        return false;
    } else
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogInfoI("Shared EGL context created ok " + std::string(errorMessage));
    }

    // ------------------------------------------------------------
    // Create window surface using the same display/config
    // ------------------------------------------------------------
    m_eglSurface = eglCreateWindowSurface(m_eglDisplay, m_eglConfig,
                                          (EGLNativeWindowType) sysInfo.info.x11.window, NULL);
    if (m_eglSurface == EGL_NO_SURFACE)
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogError("Failed to create EGL window surface" + std::string(errorMessage));
        SDL_DestroyWindow(m_window);
        SDL_Quit();
        return false;
    } else
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogInfoI("EGL window surface created ok " + std::string(errorMessage));
    }

    if (!eglMakeCurrent(m_eglDisplay, m_eglSurface, m_eglSurface, m_eglContext))
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogError("Failed to make shared EGL current" + std::string(errorMessage));
        eglDestroySurface(m_eglDisplay, m_eglSurface);
        eglDestroyContext(m_eglDisplay, m_eglContext);
        SDL_DestroyWindow(m_window);
        SDL_Quit();
        return false;
    } else
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogInfoI("Shared EGL made current ok " + std::string(errorMessage));
    }

    eglSwapInterval(m_eglDisplay, 1);

    if (!gladLoadGLES2Loader((GLADloadproc) eglGetProcAddress))
    {
        Logger::LogError("Failed to initialize GLAD");
    }

    setUICallBacks();

    return true;
}

bool GuiWindow::initializeWindow(EGLContext sharedContext)
{
    const EGLint configAttribs[] =
    {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_NONE
    };

    EGLint contextAttribs[] =
    {
        EGL_CONTEXT_MAJOR_VERSION_KHR, 3,
        EGL_NONE
    };


    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        Logger::LogError("Unable to initialize SDL: " + std::string(SDL_GetError()));
        return false;
    } else
    {
        Logger::LogInfoI("Initialized SDL" + std::string(SDL_GetError()));
    }

    int displayIndex = 0;
    SDL_Rect displayBounds;
    if (SDL_GetDisplayBounds(displayIndex, &displayBounds) < 0)
    {
        Logger::LogError("Failed to get display bounds");
    }

    m_window = SDL_CreateWindow(m_title.c_str(), displayBounds.x + m_xOffset, displayBounds.y + m_yOffset, m_width,m_height, SDL_WINDOW_OPENGL);
    if (m_window == NULL)
    {
        Logger::LogError("Unable to create window SDL: " + std::string(SDL_GetError()));
        SDL_Quit();
        return false;
    } else
    {
        Logger::LogInfoI("Created window SDL" + std::string(SDL_GetError()));
    }


    m_eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (m_eglDisplay == EGL_NO_DISPLAY)
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogError("Failed to get EGL display" + std::string(errorMessage));
        SDL_DestroyWindow(m_window);
        SDL_Quit();
        return false;
    } else
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogInfoI("EGL display ok " + std::string(errorMessage));
    }

    if (!eglInitialize(m_eglDisplay, NULL, NULL))
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogError("Failed to initialize EGL" + std::string(errorMessage));
        eglTerminate(m_eglDisplay);
        SDL_DestroyWindow(m_window);
        SDL_Quit();
        return false;
    } else
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogInfoI("EGL initialized ok " + std::string(errorMessage));
    }


    EGLint numConfigs;
    if (!eglChooseConfig(m_eglDisplay, configAttribs, &m_eglConfig, 1, &numConfigs))
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogError("Failed to choose EGL config" + std::string(errorMessage));
        eglTerminate(m_eglDisplay);
        SDL_DestroyWindow(m_window);
        SDL_Quit;
        return false;
    } else
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogInfoI("EGL config chosen ok " + std::string(errorMessage));
    }


    SDL_SysWMinfo sysInfo;
    SDL_VERSION(&sysInfo.version); // Set SDL version
    SDL_GetWindowWMInfo(m_window, &sysInfo);

    if (!eglBindAPI(EGL_OPENGL_ES_API))
    {
        EGLint error = eglGetError();
        Logger::LogError("eglBindAPI(OpenGL ES) failed: " + std::to_string(error));
        return false;
    }


    m_eglContext = eglCreateContext(m_eglDisplay, m_eglConfig, sharedContext, contextAttribs);
    if (m_eglContext == EGL_NO_CONTEXT)
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogError("Failed to create EGL context" + std::string(errorMessage));
        eglTerminate(m_eglDisplay);
        SDL_DestroyWindow(m_window);
        SDL_Quit();
        return false;
    } else
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogInfoI("EGL context created ok " + std::string(errorMessage));
    }
    m_eglSurface = eglCreateWindowSurface(m_eglDisplay, m_eglConfig, (EGLNativeWindowType) sysInfo.info.x11.window,
                                          NULL);
    //m_eglSurface = eglCreateWindowSurface(m_eglDisplay, m_eglConfig,reinterpret_cast<EGLNativeWindowType>(m_window), NULL);
    if (m_eglSurface == EGL_NO_SURFACE)
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogError("Failed to create EGL window surface" + std::string(errorMessage));
        eglTerminate(m_eglDisplay);
        SDL_DestroyWindow(m_window);
        SDL_Quit();
        return false;
    } else
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogInfoI("EGL window surface created ok " + std::string(errorMessage));
    }

    if (!eglMakeCurrent(m_eglDisplay, m_eglSurface, m_eglSurface, m_eglContext))
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogError("Failed to make EGL current" + std::string(errorMessage));
        eglDestroySurface(m_eglDisplay, m_eglSurface);
        eglDestroyContext(m_eglDisplay, m_eglContext);
        eglTerminate(m_eglDisplay);
        SDL_DestroyWindow(m_window);
        SDL_Quit();
        return false;
    } else
    {
        EGLint error = eglGetError();
        const char *errorMessage = eglGetErrorString(error);
        Logger::LogInfoI("EGL made current ok " + std::string(errorMessage));
    }

    eglSwapInterval(m_eglDisplay, 1);

    if (!gladLoadGLES2Loader((GLADloadproc) eglGetProcAddress))
    {
        Logger::LogError("Failed to initialize GLAD");
    }

    setUICallBacks();



    return true;
}

void GuiWindow::setUICallBacks()
{
    //if using EGL + SDL
#ifndef USE_EGL_SDL
    //else using GLFW
	glfwSetMouseButtonCallback(m_windowGLFW, [](GLFWwindow* window, int button, int action, int mods)
	{
			WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
			MouseButtonPressedUIEvent event((float)0, (float)0);
			data.eventCallBack(event);
	
	});
	
	
	glfwSetCursorPosCallback(m_windowGLFW, [](GLFWwindow* window, double xPos, double yPos)
	{

		WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

		MouseMovedUIEvent event((float)xPos, (float)yPos);
		data.eventCallBack(event);
		
		
		//// (1) ALWAYS forward mouse data to ImGui! This is automatic with default backends. With your own backend:
		//ImGuiIO& io = ImGui::GetIO();
		////io.AddMouseButtonEvent(button, down);
		//io.AddMousePosEvent(xPos, yPos);

		//// (2) ONLY forward mouse data to your underlying app/game.
		//if (!io.WantCaptureMouse)
		//{
		//	WindowData& data = *(WindowData*)glfwGetWindowUserPointer(GuiWindow);

		//	MouseMovedUIEvent event((float)xPos, (float)yPos);
		//	data.eventCallBackFunction(event);
		//}

	});
	
	glfwSetWindowSizeCallback(m_windowGLFW, [](GLFWwindow* window, int width, int height)
	{
		WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

		WindowResizeUIEvent event(width, height);
		data.eventCallBack(event);
	});

	glfwSetWindowCloseCallback(m_windowGLFW, [](GLFWwindow* window)
	{
		WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
		WindowCloseUIEvent event;
		data.eventCallBack(event);
	});

	glfwSetKeyCallback(m_windowGLFW, [](GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

		switch (action)
		{
			case GLFW_PRESS:
			{
				KeyPressUIEvent event(key,0);
				data.eventCallBack(event);
				break;
			}
			case GLFW_RELEASE:
			{
				KeyReleaseUIEvent event(key,2);
				data.eventCallBack(event);
				break;
			}
			case GLFW_REPEAT:
			{
				KeyPressUIEvent event(key, 1);
				data.eventCallBack(event);
				break;
			}
		}
	});

	//glfwSetCharCallback(m_windowGLFW, [](GLFWwindow* GuiWindow, unsigned int keycode)
	//{
	//	WindowData& data = *(WindowData*)glfwGetWindowUserPointer(GuiWindow);

	//	KeyTypedEvent event(keycode);
	//	data.eventCallback(event);
	//});

	//glfwSetMouseButtonCallback(m_windowGLFW, [](GLFWwindow* GuiWindow, int button, int action, int mods)
	//{
	//	WindowData& data = *(WindowData*)glfwGetWindowUserPointer(GuiWindow);

	//	switch (action)
	//	{
	//	case GLFW_PRESS:
	//	{
	//		MouseButtonPressedEvent event(button);
	//		data.EventCallback(event);
	//		break;
	//	}
	//	case GLFW_RELEASE:
	//	{
	//		MouseButtonReleasedEvent event(button);
	//		data.EventCallback(event);
	//		break;
	//	}
	//	}
	//});

	//glfwSetScrollCallback(m_windowGLFW, [](GLFWwindow* GuiWindow, double xOffset, double yOffset)
	//{
	//	WindowData& data = *(WindowData*)glfwGetWindowUserPointer(GuiWindow);

	//	MouseScrolledEvent event((float)xOffset, (float)yOffset);
	//	data.EventCallback(event);
	//});
#endif
}

void GuiWindow::onUpdateWindow()
{
#ifdef USE_EGL_SDL
    eglSwapBuffers(m_eglDisplay, m_eglSurface);
#else
	glfwSwapBuffers(m_windowGLFW);
#endif
}


void GuiWindow::printVersions()
{
    const GLubyte *renderer = glGetString(GL_RENDERER);
    const GLubyte *vendor = glGetString(GL_VENDOR);
    const GLubyte *version = glGetString(GL_VERSION);
    const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

    GLint major, minor;
#ifdef GL_MAJOR_VERSION
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
#else
    // Parse version string for OpenGL ES 2.0
if (version) {
    sscanf(reinterpret_cast<const char *>(version), "OpenGL ES %d.%d", &major, &minor);
}
#endif

    printf("\n");
    printf("GL Vendor              : %s\n", vendor);
    printf("GL Renderer            : %s\n", renderer);
    printf("GL Version (string)    : %s\n", version);
    printf("GL Version (integer) : %d.%d\n", major, minor);
    printf("GLSL Version           : %s\n", glslVersion);


    //query for supported extensions of the current OpenGL implementation
    bool logExtensions = false;
    if (logExtensions)

    {
        GLint nExtensions = 0;
#ifdef GL_NUM_EXTENSIONS
        glGetIntegerv(GL_NUM_EXTENSIONS, &nExtensions);
        for (int i = 0; i < nExtensions; i++)
        {
            printf("%s\n", glGetStringi(GL_EXTENSIONS, i));
        }
#else
        const char *extensions = reinterpret_cast<const char *>(glGetString(GL_EXTENSIONS));
    if (extensions) {
        printf("%s\n", extensions);
    }
#endif
    }

    printf("\n");
}

void GuiWindow::exit()
{
    cleanup();
}

void GuiWindow::cleanup()
{
#ifdef USE_EGL_SDL
    if (m_eglDisplay != EGL_NO_DISPLAY)
    {
        eglMakeCurrent(m_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (m_eglContext != EGL_NO_CONTEXT)
        {
            eglDestroyContext(m_eglDisplay, m_eglContext);
            m_eglContext = EGL_NO_CONTEXT;
        }
        if (m_eglSurface != EGL_NO_SURFACE)
        {
            eglDestroySurface(m_eglDisplay, m_eglSurface);
            m_eglSurface = EGL_NO_SURFACE;
        }
        eglTerminate(m_eglDisplay);
        m_eglDisplay = EGL_NO_DISPLAY;
    }

    if (m_window != nullptr)
    {
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
        SDL_Quit();
    }
#else
    if(m_windowGLFW != NULL)
		glfwDestroyWindow(m_windowGLFW);
    glfwTerminate();
#endif
}
