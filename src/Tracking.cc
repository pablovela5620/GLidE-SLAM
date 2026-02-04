/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <unistd.h>

#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
//#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include "ImageHandler.h"
#include "Logger.h"
#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{
    Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, Map *pMap, KeyFrameDatabase *pKFDB,
                       const string &strSettingPath, const int sensor) : mState(NO_IMAGES_YET), mSensor(sensor),
                                                                         mbOnlyTracking(false), mbVO(false),
                                                                         mpORBVocabulary(pVoc),
                                                                         mpKeyFrameDB(pKFDB),
                                                                         mpInitializer(
                                                                             static_cast<Initializer *>(NULL)),
                                                                         mpSystem(pSys), mpViewer(NULL),
                                                                         mpMap(pMap), mnLastRelocFrameId(0)
    {
        // Load camera parameters from settings file

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3,CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        mFx = fx;
        mFy = fy;
        mCx = cx;
        mCy = cy;

        mScaleFactors.resize(NLEVELS_DIRECT);
        mScaleFactors[0] = 1.0f;
        for (int i = 1; i < NLEVELS_DIRECT; i++)
        {
            mScaleFactors[i] = mScaleFactors[i - 1] * SCALE_FACTOR;
        }

        cv::Mat DistCoef(4, 1,CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;

        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        // Load ORB parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::MONOCULAR)
            mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        if (sensor == System::STEREO || sensor == System::RGBD)
        {
            mThDepth = mbf * (float) fSettings["ThDepth"] / fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }

        if (sensor == System::RGBD)
        {
            mDepthMapFactor = fSettings["DepthMapFactor"];
            if (fabs(mDepthMapFactor) < 1e-5)
                mDepthMapFactor = 1;
            else
                mDepthMapFactor = 1.0f / mDepthMapFactor;
        }


        const std::string indirectTweenFrame = "indFrames.txt";
        const std::string directTweenFrame = "dFrames.txt";

        std::ofstream(indirectTweenFrame.c_str(), std::ios::out | std::ios::trunc).close();
        std::ofstream(directTweenFrame.c_str(), std::ios::out | std::ios::trunc).close();
    }

    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
    {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
    {
        mpLoopClosing = pLoopClosing;
    }

    void Tracking::SetViewer(Viewer *pViewer)
    {
        mpViewer = pViewer;
    }

    cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
    {
        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;

        if (mImGray.channels() == 3)
        {
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
                cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGB2GRAY);
            } else
            {
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
                cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGR2GRAY);
            }
        } else if (mImGray.channels() == 4)
        {
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
                cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGBA2GRAY);
            } else
            {
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
                cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGR2GRAY);
            }
        }

        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary,
                              mK, mDistCoef, mbf, mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp)
    {
        mImGray = imRGB;
        cv::Mat imDepth = imD;

        if (mImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
        } else if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
        }

        if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
            imDepth.convertTo(imDepth,CV_32F, mDepthMapFactor);

        mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf,
                              mThDepth);

        Track();

        return mCurrentFrame.mTcw.clone();
    }

    cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
    {
        mImGray = im;
        if (mImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
        } else if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
        }


        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
        else
        {
            mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf,
                                  mThDepth);
            mCurrentDirectFrame = FrameDirect(mImGray, timestamp, mK, mDistCoef);
        }

        Logger<std::string>::LogInfoII("\n Input frame: " + std::to_string(mCurrentFrame.mnId));

        Track();


        //write csv file to compare to ground-truth:
        std::string indirectTweenFrame = "indFrames.txt";
        std::string directTweenFrame = "dFrames.txt";

        //Logger<std::string>::LogInfoII("\n Timestamp: " + to_string(timestamp));
        if (mTweenFrameData.size() > 0) WriteTweenFrameData(indirectTweenFrame, mTweenFrameData, mCurrentFrame.mnId);
        if (mDTweenFrameData.size() > 0) WriteTweenFrameData(directTweenFrame, mDTweenFrameData, mCurrentFrame.mnId);


        return mCurrentFrame.mTcw.clone();
    }


    // GLidE-SLAM Track()
    void Tracking::Track()
    {
        if (mState == NO_IMAGES_YET)
        {
            mState = NOT_INITIALIZED;
        }

        mLastProcessedState = mState;

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        if (mState == NOT_INITIALIZED)
        {
            if (mSensor == System::STEREO || mSensor == System::RGBD)
                StereoInitialization();
            else
                MonocularInitialization();

            //mpFrameDrawer->Update(this);

            if (mState != OK)
                return;
        } else
        {
            // System is initialized. Track Frame.
            bool bOK;

            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            if (!mbOnlyTracking)
            {
                // Local Mapping is activated. This is the normal behaviour, unless
                // you explicitly activate the "only tracking" mode.

                if (mState == OK)
                {
                    // Local Mapping might have changed some MapPoints tracked in last frame
                    CheckReplacedInLastFrame();

                    bool bDirectTrackRecovery = mCurrentDirectFrame.mnId < mpPrevDirectRefID + 3;
                    mbDirectTrackOk = trackDirectIC(&mCurrentDirectFrame, &mLastDirectFrame, m_directTrackCache,
                                                        false, mLastDirectChi2);
                    bool bSwitchToIndirect = SwitchToIndirect(mLastDirectChi2);
                    mbUseDirectTracking = false;
                    if (mbDirectTrackOk && !bSwitchToIndirect)
                    {
                        // Tween frame: use direct pose, skip TrackLocalMap
                        mCurrentFrame.SetPose(mCurrentDirectFrame.mTcw);
                        mbUseDirectTracking = true;
                        bOK = true;
                        mpMap->AddDirectTweenFrame(mCurrentDirectFrame);
                        mpMap->NotifyFramesUpdated();
                    }
                    else
                    {
                        if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                        {
                            bOK = TrackReferenceKeyFrame();
                        } else
                        {
                            bOK = TrackWithMotionModel();
                            if (!bOK)
                                bOK = TrackReferenceKeyFrame();
                        }
                    }
                }
                else
                {
                    bOK = Relocalization();
                }
            }
            else
            {
                // Localization Mode: Local Mapping is deactivated

                if (mState == LOST)
                {
                    bOK = Relocalization();
                } else
                {
                    if (!mbVO)
                    {
                        // In last frame we tracked enough MapPoints in the map

                        if (!mVelocity.empty())
                        {
                            bOK = TrackWithMotionModel();
                        } else
                        {
                            bOK = TrackReferenceKeyFrame();
                        }
                    } else
                    {
                        // In last frame we tracked mainly "visual odometry" points.

                        // We compute two camera poses, one from motion model and one doing relocalization.
                        // If relocalization is sucessfull we choose that solution, otherwise we retain
                        // the "visual odometry" solution.

                        bool bOKMM = false;
                        bool bOKReloc = false;
                        vector<MapPoint *> vpMPsMM;
                        vector<bool> vbOutMM;
                        cv::Mat TcwMM;
                        if (!mVelocity.empty())
                        {
                            bOKMM = TrackWithMotionModel();
                            vpMPsMM = mCurrentFrame.mvpMapPoints;
                            vbOutMM = mCurrentFrame.mvbOutlier;
                            TcwMM = mCurrentFrame.mTcw.clone();
                        }
                        bOKReloc = Relocalization();

                        if (bOKMM && !bOKReloc)
                        {
                            mCurrentFrame.SetPose(TcwMM);
                            mCurrentFrame.mvpMapPoints = vpMPsMM;
                            mCurrentFrame.mvbOutlier = vbOutMM;

                            if (mbVO)
                            {
                                for (int i = 0; i < mCurrentFrame.N; i++)
                                {
                                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                    {
                                        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                    }
                                }
                            }
                        } else if (bOKReloc)
                        {
                            mbVO = false;
                        }

                        bOK = bOKReloc || bOKMM;
                    }
                }
            }

            mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if (!mbOnlyTracking)
            {
                if (bOK)
                {
                    //add VO frame for visualization
                    if (mbUseDirectTracking)
                    {
                        std::vector<double> directTweenFrameData;
                        FetchPosandRot(mCurrentDirectFrame.mTimeStamp, mCurrentDirectFrame.mRwc,
                                       mCurrentDirectFrame.mtwc, directTweenFrameData);
                        mDTweenFrameData = directTweenFrameData;
                    } else
                    {
                        std::vector<double> indirectTweenFrameData;
                        FetchPosandRot(mCurrentFrame.mTimeStamp, mCurrentFrame.mRwc, mCurrentFrame.mtwc,
                                       indirectTweenFrameData);
                        mTweenFrameData = indirectTweenFrameData;
                        bOK = TrackLocalMap();
                    }
                }
            }
            else
            {
                // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
                // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
                // the camera we will use the local map again.
                if (bOK && !mbVO)
                    bOK = TrackLocalMap();
            }

            if (bOK)
                mState = OK;
            else
                mState = LOST;

            // Update drawer
            //mpFrameDrawer->Update(this);


            if (mbDirectTrackOk && mbUseDirectTracking)
            {
                if (!mLastDirectFrame.mTcw.empty())
                {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4,CV_32F);
                    mLastDirectFrame.mRwc.copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastDirectFrame.mtwc.copyTo(LastTwc.rowRange(0, 3).col(3));
                    mVelocityDirect = mCurrentDirectFrame.mTcw * LastTwc;
                } else
                    mVelocityDirect = cv::Mat();
            } else
            {
                mVelocityDirect = cv::Mat();
                updateDirectReference();
            }


            // If tracking were good, check if we insert a keyframe
            if (!mbUseDirectTracking)
            {
                if (bOK)
                {
                    // Update motion model
                    if (!mLastFrame.mTcw.empty())
                    {
                        cv::Mat LastTwc = cv::Mat::eye(4, 4,CV_32F);
                        mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                        mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                        mVelocity = mCurrentFrame.mTcw * LastTwc;
                    } else
                        mVelocity = cv::Mat();


                    //mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

                    // Clean VO matches
                    for (int i = 0; i < mCurrentFrame.N; i++)
                    {
                        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                        if (pMP)
                            if (pMP->Observations() < 1)
                            {
                                mCurrentFrame.mvbOutlier[i] = false;
                                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                            }
                    }

                    // Delete temporal MapPoints
                    for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
                         lit != lend; lit++)
                    {
                        MapPoint *pMP = *lit;
                        delete pMP;
                    }
                    mlpTemporalPoints.clear();

                    // Check if we need to insert a new keyframe
                    if (NeedNewKeyFrame())
                    {
                        CreateNewKeyFrame();
                    }
                    //only for direct tracking
                    else
                    {
                        if (NeedNewDirectRef());
                    }

                    // We allow points with high innovation (considererd outliers by the Huber Function)
                    // pass to the new keyframe, so that bundle adjustment will finally decide
                    // if they are outliers or not. We don't want next frame to estimate its position
                    // with those points so we discard them in the frame.
                    for (int i = 0; i < mCurrentFrame.N; i++)
                    {
                        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
                }

                // Reset if the camera get lost soon after initialization
                if (mState == LOST)
                {
                    if (mpMap->KeyFramesInMap() <= 5)
                    {
                        cout << "Track lost soon after initialisation, reseting..." << endl;
                        mpSystem->Reset();
                        return;
                    }
                }

                if (!mCurrentFrame.mpReferenceKF)
                    mCurrentFrame.mpReferenceKF = mpReferenceKF;

                mLastFrame = Frame(mCurrentFrame);
            }

            if (mbDirectTrackOk)
                mLastDirectFrame = FrameDirect(mCurrentDirectFrame);
        }

        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if (!mCurrentFrame.mTcw.empty())
        {
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        } else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }
    }

    // NEW: trackDirectIC with "patch correction" (local search) + discarding
    // - Pass 1 (iter==0 only): for each point, if projected patch MSE is bad -> search small neighborhood for best match,
    //   discard if still bad / too far / ambiguous. Store (du,dv,valid) per point.
    // - Pass 2 (all iters): reproject with current pose each iter, then sample patch at (uc+du, vc+dv) and accumulate H,b.
    bool Tracking::trackDirectIC(FrameDirect *newFrame, FrameDirect *oldFrame,
                                     const std::vector<DirectTrackCache> &dtCache, bool useMotion, float &chi2)
    {
        if (useMotion)
            newFrame->SetPose(mVelocityDirect * oldFrame->mTcw);
        else
            newFrame->SetPose(oldFrame->mTcw);

        cv::Mat Tcw = newFrame->mTcw;

        Logger<std::string>::LogInfoIII(
            "Direct Tracker: Frames: " + std::to_string(newFrame->mnId) + " - " + std::to_string(mpPrevDirectRefID) +
            " levels=" + std::to_string(dtCache.size()) +
            " useMotion=" + std::to_string((int) useMotion));

        bool completeLog = false;

        bool anyConverged = false;
        float finalChi2 = std::numeric_limits<float>::max();

        const float huberK = 0.08f;

        // NEW: patch search parameters (tune later)
        const int searchRadius = 3; // NEW: +/- pixels
        const float ambigRatioThresh = 0.95f; // NEW: best/second-best too close => ambiguous => discard

        // NEW: per-level thresholds (reasonable starting points for PATCH_SIZE=11)
        // Use index clamp if you have >4 levels.
        const float searchThresholdL[4] = {0.012f, 0.018f, 0.025f, 0.035f}; // NEW: if projected MSE > this => search
        const float rejectThresholdL[4] = {0.025f, 0.035f, 0.050f, 0.070f}; // NEW: if best MSE > this => discard
        const int maxShiftL[4] = {3, 4, 5, 6}; // NEW: if shift > this => discard

        for (int level = (int) dtCache.size() - 1; level >= 0; --level)
        {
            bool hadValidIteration = false;
            bool acceptedStep = false;

            const auto &levelCache = dtCache[level];
            const cv::Mat &InewFrame = newFrame->m_pyrImg[level];
            if (InewFrame.empty())
            {
                Logger<std::string>::LogError("Direct Tracker: Could not track, I new is empty");
                return false;
            }

            float invs = 1.0f / mScaleFactors[level];
            float fx = mFx * invs;
            float fy = mFy * invs;
            float cx = mCx * invs;
            float cy = mCy * invs;

            int border = static_cast<int>(PATCH_CENTER) + 2;
            int borderSearch = border + searchRadius; // NEW: extra margin for search window

            float lastChi2 = std::numeric_limits<float>::max();
            float bestChi2 = std::numeric_limits<float>::max();
            cv::Mat bestTcw = Tcw.clone();
            bool converged = false;
            int divergeCount = 0;

            // ============================================================
            // NEW: per-level stored correction (du,dv) and validity per point
            // Computed once at iter==0 and reused for all iterations.
            // ============================================================
            struct Align2DResult
            {
                int8_t du; // NEW
                int8_t dv; // NEW
                uint8_t valid; // NEW
                float bestMSE; // NEW (optional, mostly for debug)
            };

            std::vector<Align2DResult> alignRes;
            alignRes.resize(levelCache.pointData.size());
            bool alignComputed = false;

            // NEW: clamp level index for threshold arrays
            const int li = (level < 0) ? 0 : (level > 3 ? 3 : level);
            const float searchThreshold = searchThresholdL[li];
            const float rejectThreshold = rejectThresholdL[li];
            const int maxShift = maxShiftL[li];

            // NEW: helpers
            auto fullPatchMSE = [&](const DirectPointData &dp, float uc0, float vc0) -> float
            {
                float sse = 0.0f;
                int k = 0;
                for (int dy = 0; dy < PATCH_SIZE; ++dy)
                {
                    for (int dx = 0; dx < PATCH_SIZE; ++dx)
                    {
                        float u = uc0 + dx - PATCH_CENTER;
                        float v = vc0 + dy - PATCH_CENTER;
                        float Ic = ImageHandler::bilinearInterpolation(InewFrame, u, v);
                        float res = dp.I[k] - Ic;
                        sse += res * res;
                        ++k;
                    }
                }
                return sse / (float) (PATCH_SIZE * PATCH_SIZE);
            };

            // NEW: cheap MSE (3x3 samples inside patch) for gating/search scoring
            auto coarseMSE3x3 = [&](const DirectPointData &dp, float uc0, float vc0) -> float
            {
                int step = (int) PATCH_CENTER - 1;
                if (step < 1) step = 1;
                const int offs[3] = {-step, 0, step};

                float sse = 0.0f;
                int cnt = 0;

                for (int iy = 0; iy < 3; ++iy)
                {
                    for (int ix = 0; ix < 3; ++ix)
                    {
                        int dx = offs[ix];
                        int dy = offs[iy];

                        int px = dx + (int) PATCH_CENTER;
                        int py = dy + (int) PATCH_CENTER;
                        int k = py * (int) PATCH_SIZE + px;

                        float u = uc0 + (float) dx;
                        float v = vc0 + (float) dy;

                        float Ic = ImageHandler::bilinearInterpolation(InewFrame, u, v);
                        float res = dp.I[k] - Ic;

                        sse += res * res;
                        ++cnt;
                    }
                }

                return (cnt > 0) ? (sse / (float) cnt) : std::numeric_limits<float>::max();
            };

            auto searchBestOffsetCoarse = [&](const DirectPointData &dp, float uc0, float vc0,
                                              int &outDu, int &outDv,
                                              float &bestScore, float &secondBestScore) -> void
            {
                bestScore = std::numeric_limits<float>::max();
                secondBestScore = std::numeric_limits<float>::max();
                outDu = 0;
                outDv = 0;

                for (int dv = -searchRadius; dv <= searchRadius; ++dv)
                {
                    for (int du = -searchRadius; du <= searchRadius; ++du)
                    {
                        float score = coarseMSE3x3(dp, uc0 + (float) du, vc0 + (float) dv);

                        if (score < bestScore)
                        {
                            secondBestScore = bestScore;
                            bestScore = score;
                            outDu = du;
                            outDv = dv;
                        } else if (score < secondBestScore)
                        {
                            secondBestScore = score;
                        }
                    }
                }
            };
            // ============================================================

            for (int iter = 0; iter < 10; ++iter)
            {
                cv::Mat R = Tcw.rowRange(0, 3).colRange(0, 3);
                cv::Mat t = Tcw.rowRange(0, 3).col(3);

                Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
                //Eigen::Matrix<float,6,6> H = Eigen::Matrix<float,6,6>::Zero();

                Eigen::Matrix<float, 6, 6> H = levelCache.H;

                float chi2 = 0.0f;
                int n = 0;
                int nPointsUsed = 0;
                int nOOB = 0;
                int nNegZ = 0;
                int nDiscard = 0;

                float sumSignedRes = 0.0f;
                int convergenceIter = -1;

                // ============================================================
                // NEW (PASS 1): compute (du,dv) + valid ONLY ONCE at iter==0
                // ============================================================
                if (!alignComputed)
                {
                    int nSearched = 0;
                    int nValid = 0;
                    float sumShift = 0.0f;
                    float worstShift = 0.0f;

                    for (size_t i = 0; i < levelCache.pointData.size(); ++i)
                    {
                        const auto &dp = levelCache.pointData[i];

                        Align2DResult ar;
                        ar.du = 0;
                        ar.dv = 0;
                        ar.valid = 0;
                        ar.bestMSE = std::numeric_limits<float>::max();

                        cv::Mat Pc = R * dp.mapPoint->GetWorldPos() + t;
                        float Zc = Pc.at<float>(2);
                        if (Zc <= 0.0f)
                        {
                            alignRes[i] = ar;
                            continue;
                        }

                        float uc = fx * Pc.at<float>(0) / Zc + cx;
                        float vc = fy * Pc.at<float>(1) / Zc + cy;

                        // Need extra border for search
                        if (uc < borderSearch || vc < borderSearch ||
                            uc >= InewFrame.cols - borderSearch ||
                            vc >= InewFrame.rows - borderSearch)
                        {
                            alignRes[i] = ar;
                            continue;
                        }

                        // Cheap score at projection
                        float mseProjCoarse = coarseMSE3x3(dp, uc, vc);

                        int bestDu = 0, bestDv = 0;
                        float bestScore = mseProjCoarse;
                        float secondBest = std::numeric_limits<float>::max();

                        if (mseProjCoarse > searchThreshold)
                        {
                            // Search (coarse scoring) only for bad projections
                            ++nSearched;
                            searchBestOffsetCoarse(dp, uc, vc, bestDu, bestDv, bestScore, secondBest);
                        }

                        float shift = std::sqrt((float) (bestDu * bestDu + bestDv * bestDv));

                        // Ambiguity check from coarse scores
                        bool ambiguous = false;
                        if (std::isfinite(secondBest) && secondBest > 1e-12f)
                        {
                            float ratio = bestScore / secondBest;
                            if (ratio > ambigRatioThresh)
                                ambiguous = true;
                        }

                        // Compute full-patch MSE only at the best candidate (fast + correct gating)
                        float bestMSE = fullPatchMSE(dp, uc + (float) bestDu, vc + (float) bestDv);

                        // Discard rules
                        if (ambiguous || shift > (float) maxShift || bestMSE > rejectThreshold)
                        {
                            ar.valid = 0;
                        } else
                        {
                            ar.du = (int8_t) bestDu;
                            ar.dv = (int8_t) bestDv;
                            ar.valid = 1;
                            ar.bestMSE = bestMSE;

                            ++nValid;
                            sumShift += shift;
                            if (shift > worstShift) worstShift = shift;
                        }

                        alignRes[i] = ar;
                    }

                    // Log once per level at iter==0
                    if (completeLog)
                    {
                        Logger<std::string>::LogInfoI(
                            "ALIGN: L=" + std::to_string(level) +
                            " valid=" + std::to_string(nValid) +
                            " searched=" + std::to_string(nSearched) +
                            " avgShift=" + std::to_string(nValid > 0 ? (sumShift / (float) nValid) : 0.0f) +
                            " worstShift=" + std::to_string(worstShift) +
                            " searchTh=" + std::to_string(searchThreshold) +
                            " rejectTh=" + std::to_string(rejectThreshold) +
                            " maxShift=" + std::to_string(maxShift));
                    }
                    alignComputed = true;
                }
                // ============================================================

                // ============================================================
                // PASS 2: use stored offsets; reproject every iteration
                // ============================================================
                for (size_t i = 0; i < levelCache.pointData.size(); ++i)
                {
                    const auto &dp = levelCache.pointData[i];
                    const Align2DResult &ar = alignRes[i];

                    if (!ar.valid)
                    {
                        ++nDiscard;
                        continue;
                    }

                    cv::Mat Pc = R * dp.mapPoint->GetWorldPos() + t;
                    float Zc = Pc.at<float>(2);
                    if (Zc <= 0.0f)
                    {
                        ++nNegZ;
                        continue;
                    }

                    float uc = fx * Pc.at<float>(0) / Zc + cx;
                    float vc = fy * Pc.at<float>(1) / Zc + cy;

                    // NEW: corrected sampling center
                    float uc2 = uc + (float) ar.du;
                    float vc2 = vc + (float) ar.dv;

                    // Border check for corrected patch center
                    if (uc2 < border || vc2 < border ||
                        uc2 >= InewFrame.cols - border ||
                        vc2 >= InewFrame.rows - border)
                    {
                        ++nOOB;
                        continue;
                    }

                    ++nPointsUsed;

                    int k = 0;
                    for (int dy = 0; dy < PATCH_SIZE; ++dy)
                    {
                        for (int dx = 0; dx < PATCH_SIZE; ++dx)
                        {
                            float u = uc2 + dx - PATCH_CENTER;
                            float v = vc2 + dy - PATCH_CENTER;

                            float Ic = ImageHandler::bilinearInterpolation(InewFrame, u, v);
                            float res = dp.I[k] - Ic;

                            sumSignedRes += res;

                            float w = (std::fabs(res) <= huberK) ? 1.0f : (huberK / std::fabs(res));

                            //H += w * (dp.J[k].transpose() * dp.J[k]);
                            //H += w * (dp.J[k].transpose() * dp.J[k]);
                            b += w * dp.J[k].transpose() * res;
                            chi2 += w * res * res;

                            ++k;
                            ++n;
                        }
                    }
                }
                // ============================================================

                if (n < 16 * 3)
                {
                    Logger<std::string>::LogError(
                        "Direct: L=" + std::to_string(level) +
                        " iter=" + std::to_string(iter) +
                        " too few meas=" + std::to_string(n) +
                        " pts=" + std::to_string(nPointsUsed) +
                        " OOB=" + std::to_string(nOOB) +
                        " negZ=" + std::to_string(nNegZ) +
                        " disc=" + std::to_string(nDiscard));
                    break;
                }

                if (H.diagonal().minCoeff() < 1e-6f)
                {
                    Logger<std::string>::LogError(
                        "DirectIC: L=" + std::to_string(level) +
                        " singular H, minDiag=" + std::to_string(H.diagonal().minCoeff()));
                    break;
                }

                float chi2Mean = chi2 / (float) n;

                if (completeLog)
                {
                    Logger<std::string>::LogInfoI(
                        "DirectIC: L=" + std::to_string(level) +
                        " iter=" + std::to_string(iter) +
                        " chi2=" + std::to_string(chi2Mean) +
                        " pts=" + std::to_string(nPointsUsed) +
                        " meas=" + std::to_string(n) +
                        " disc=" + std::to_string(nDiscard));
                }

                if (chi2Mean < bestChi2)
                {
                    bestChi2 = chi2Mean;
                    bestTcw = Tcw.clone();
                    divergeCount = 0;
                } else
                {
                    ++divergeCount;
                    if (divergeCount >= 3)
                    {
                        if (hadValidIteration)
                        {
                            Tcw = bestTcw.clone();
                            acceptedStep = true;
                            converged = true;
                        }
                        break;
                    }
                }

                float meanSignedRes = sumSignedRes / (float) n;

                if (iter == 0 && level == 0)
                {
                    Logger<std::string>::LogInfoIII(
                        "DIAG: meanSignedRes=" + std::to_string(meanSignedRes) +
                        " chi2=" + std::to_string(chi2Mean));
                }

                lastChi2 = chi2Mean;

                Eigen::Matrix<float, 6, 1> delta = H.ldlt().solve(b);
                //Eigen::Matrix<float,6,1> delta = -H.ldlt().solve(b);


                if (!delta.allFinite())
                {
                    Logger<std::string>::LogError(
                        "DirectIC: L=" + std::to_string(level) +
                        " iter=" + std::to_string(iter) +
                        " delta not finite");
                    break;
                }

                hadValidIteration = true;

                cv::Matx<float, 6, 1> xi(delta(3), delta(4), delta(5), delta(0), delta(1), delta(2));

                Tcw = Tcw * cv::Mat(ImageHandler::se3exp(xi));
                //Tcw = Tcw * cv::Mat(ImageHandler::se3exp(xi));


                if (delta.norm() < 1e-4f)
                {
                    acceptedStep = true;
                    converged = true;
                    break;
                }

                // if (iter == 2)
                //     alignComputed = false;
                // OPTIONAL NEW: if you want a cheap "refresh", do it ONCE later (still much cheaper than every iter)
                // if (iter == 3) alignComputed = false;
            }

            if (acceptedStep)
            {
                anyConverged = true;
            } else if (hadValidIteration)
            {
                Tcw = bestTcw.clone();
                Logger<std::string>::LogWarning(
                    "DirectIC: L=" + std::to_string(level) +
                    " best-effort pose (no convergence), bestChi2=" + std::to_string(bestChi2));
            } else
            {
                Logger<std::string>::LogWarning(
                    "DirectIC: L=" + std::to_string(level) +
                    " failed (no valid iteration).");
            }

            if (!converged)
            {
                Logger<std::string>::LogWarning(
                    "DirectIC: L=" + std::to_string(level) +
                    " max iter reached, chi2=" + std::to_string(lastChi2));
            }

            if (level == 0)
                finalChi2 = bestChi2;

            //collect inliers (only do this if reach level 0)
            if (level == 0)
            {
                const DirectTrackCache &levelCache0 = dtCache[level];
                const float mseThreshold = 3.0f * finalChi2;
                trackDirectInliers(levelCache0, Tcw, mseThreshold);
            }
        }

        chi2 = finalChi2;
        if (!anyConverged)
        {
            Logger<std::string>::LogError("DirectIC: REJECT,  no level converged");
            return false;
        }

        if (finalChi2 > 0.0045f)
        {
            Logger<std::string>::LogError("DirectIC: REJECT, chi2 too high (" + std::to_string(finalChi2) + ")");
            return false;
        }

        Logger<std::string>::LogInfoIII("DirectIC: SUCCESS, final chi2: " + std::to_string(finalChi2));
        newFrame->SetPose(Tcw);
        return true;
    }

    bool Tracking::trackPrecompute(const Frame &frame, std::vector<DirectTrackCache> &dtCache)
    {
        Logger<std::string>::LogInfoIII(
            "Direct Tracker: Precompute IC on reference frame: " + std::to_string(frame.mnId));

        //static const int nLevels = 4;
        dtCache.resize(NLEVELS_DIRECT);

        for (int level = 0; level < NLEVELS_DIRECT; ++level)
        {
            auto &L = dtCache[level];
            L.H.setZero();
            L.pointData.clear();

            const cv::Mat &Iref = frame.m_pyrImg[level];
            if (Iref.empty())
            {
                Logger<std::string>::LogError("Direct Tracker: Level " + std::to_string(level) + " image empty");
                return false;
            }

            float invs = 1.0f / mScaleFactors[level];
            float fx = mFx * invs;
            float fy = mFy * invs;
            float cx = mCx * invs;
            float cy = mCy * invs;
            int border = static_cast<int>(PATCH_CENTER) + 2;

            cv::Mat Rc = frame.mRcw;
            cv::Mat tc = frame.mtcw;

            for (MapPoint *mp: frame.mvpMapPoints)
            {
                if (!mp || mp->isBad()) continue;

                cv::Mat Pc = Rc * mp->GetWorldPos() + tc;
                float Xr = Pc.at<float>(0);
                float Yr = Pc.at<float>(1);
                float Zr = Pc.at<float>(2);
                if (Zr <= 0) continue;

                float ur = fx * Xr / Zr + cx;
                float vr = fy * Yr / Zr + cy;
                if (ur < border || vr < border ||
                    ur >= Iref.cols - border || vr >= Iref.rows - border)
                    continue;

                DirectPointData dp;
                dp.mapPoint = mp;

                float invZ = 1.0f / Zr, invZ2 = invZ * invZ;
                int k = 0;

                for (int dy = 0; dy < PATCH_SIZE; ++dy)
                {
                    for (int dx = 0; dx < PATCH_SIZE; ++dx)
                    {
                        float u = ur + dx - PATCH_CENTER;
                        float v = vr + dy - PATCH_CENTER;

                        float gx = 0.5f * (ImageHandler::bilinearInterpolation(Iref, u + 1, v) -
                                           ImageHandler::bilinearInterpolation(Iref, u - 1, v));
                        float gy = 0.5f * (ImageHandler::bilinearInterpolation(Iref, u, v + 1) -
                                           ImageHandler::bilinearInterpolation(Iref, u, v - 1));

                        Eigen::Matrix<float, 1, 6> J;
                        J << gx * fx * invZ,
                                gy * fy * invZ,
                                -(gx * fx * Xr + gy * fy * Yr) * invZ2,
                                -gx * fx * Xr * Yr * invZ2 - gy * fy * (1 + Yr * Yr * invZ2),
                                gx * fx * (1 + Xr * Xr * invZ2) + gy * fy * Xr * Yr * invZ2,
                                (-gx * fx * Yr + gy * fy * Xr) * invZ;

                        dp.J[k] = J;
                        dp.I[k] = ImageHandler::bilinearInterpolation(Iref, u, v);
                        L.H += J.transpose() * J;
                        ++k;
                    }
                }
                L.pointData.push_back(dp);
            }

            if (L.pointData.size() < 20)
            {
                Logger<std::string>::LogError(
                    "Direct Tracker: Level " + std::to_string(level) + " too few points: " + std::to_string(
                        L.pointData.size()));
                return false;
            }
        }
        return true;
    }

    bool Tracking::SwitchToIndirect(float chi2)
    {
        // Check if local mapping is busy
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
            return false;

        const int nKFs = mpMap->KeyFramesInMap();

        // Do not insert keyframes if not enough frames have passed
        if (mCurrentDirectFrame.mnId < mpPrevDirectRefID + mMaxFramesDirect && nKFs > mMaxFramesDirect)
            return false;

        // --- Temporal conditions (from original) ---
        const bool c1a = mCurrentDirectFrame.mnId >= mpPrevDirectRefID + mMaxFramesDirect;
        const bool c1b = (mCurrentDirectFrame.mnId >= mpPrevDirectRefID + mMinFrames &&
                          !mpLocalMapper->KeyframesInQueue());

        // --- Chi2-based quality conditions (replaces map point stats) ---
        const bool bTrackingWeak = (chi2 > 0.004f); // Emergency: tracking degrading fast
        const bool bQualityDegrading = (chi2 > 0.0025f); // Quality dropping vs reference

        std::string c1aString = (c1a) ? "true" : "false";
        std::string c1bString = (c1b) ? "true" : "false";
        std::string c1cString = (bTrackingWeak) ? "true" : "false";
        std::string c2String = (bQualityDegrading) ? "true" : "false";


        if ((c1a || c1b || bTrackingWeak) && bQualityDegrading)
        {
            Logger<std::string>::LogInfoI("Switch to Indirect tracking: " + std::to_string(mCurrentFrame.mnId) +
                                          ", More than max frames=" + c1aString +
                                          ", More than min frames=" + c1bString +
                                          ", Tracking too weak=" + c1cString +
                                          ", Tracking degraded=" + c2String);
            mpPrevDirectRefID = mCurrentFrame.mnId;
            return true;
        }
        return false;
    }

    int Tracking::trackDirectInliers(const DirectTrackCache &cache, const cv::Mat &Tcw, const float mseThreshold)
    {
        mvpLocalDirectInliers.clear();
        mvpLocalDirectInliers.reserve(cache.pointData.size());

        cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
        const cv::Mat &I = mCurrentDirectFrame.m_pyrImg[0];

        int border = PATCH_CENTER + 2;

        for (const auto &dp: cache.pointData)
        {
            if (!dp.mapPoint || dp.mapPoint->isBad()) continue;

            cv::Mat Pc = Rcw * dp.mapPoint->GetWorldPos() + tcw;
            float z = Pc.at<float>(2);
            if (z <= 0.0f) continue;

            float uc = mCurrentFrame.fx * Pc.at<float>(0) / z + mCurrentFrame.cx;
            float vc = mCurrentFrame.fy * Pc.at<float>(1) / z + mCurrentFrame.cy;

            if (uc < border || vc < border || uc >= I.cols - border || vc >= I.rows - border)
                continue;

            float sumRes2 = 0.0f;
            for (int k = 0; k < PATCH_AREA; ++k)
            {
                float u = uc + (k % PATCH_SIZE) - PATCH_CENTER;
                float v = vc + (k / PATCH_SIZE) - PATCH_CENTER;
                float res = dp.I[k] - ImageHandler::bilinearInterpolation(I, u, v);
                sumRes2 += res * res;
            }

            float mse = sumRes2 / (float) PATCH_AREA;
            if (mse > mseThreshold) continue;

            mvpLocalDirectInliers.push_back(dp.mapPoint);
        }

        Logger<std::string>::LogInfoIII("direct inliers matches: " + std::to_string(mvpLocalDirectInliers.size()));
        return (int) mvpLocalDirectInliers.size();
    }

    void Tracking::FetchPosandRot(const double timeStamp, const cv::Mat &R, const cv::Mat &t,
                                  std::vector<double> &output)
    {
        //timestamp tx ty tz qx qy qz qw
        output.clear();
        output.reserve(8);
        if (R.empty() || t.empty()) return;

        // Convert to double to avoid CV_32F/at<double> garbage
        cv::Mat R64, t64;
        R.convertTo(R64, CV_64F);
        t.convertTo(t64, CV_64F);

        double tx = t64.at<double>(0, 0);
        double ty = t64.at<double>(1, 0);
        double tz = t64.at<double>(2, 0);

        cv::Mat rvec;
        cv::Rodrigues(R64, rvec);

        double angle = cv::norm(rvec);
        double qw = std::cos(angle * 0.5);
        double s = (angle < 1e-12) ? 0.5 : (std::sin(angle * 0.5) / angle);

        double qx = rvec.at<double>(0, 0) * s;
        double qy = rvec.at<double>(1, 0) * s;
        double qz = rvec.at<double>(2, 0) * s;

        output.push_back(timeStamp);
        output.push_back(tx);
        output.push_back(ty);
        output.push_back(tz);
        output.push_back(qx);
        output.push_back(qy);
        output.push_back(qz);
        output.push_back(qw);
    }

    void Tracking::WriteTweenFrameData(std::string &path, const std::vector<double> &data,
                                       const unsigned long int frameNumber)
    {
        if (data.size() != 8) return;

        static std::mutex m;
        std::lock_guard<std::mutex> lock(m);

        std::ofstream f(path, std::ios::app);
        if (!f) return;


        f << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (i) f << ' ';
            f << data[i];
        }

        f << ' ' << frameNumber << '\n';
    }

    void Tracking::updateDirectReference()
    {
        //mpMap->ClearTweenFrames();
        mpPrevDirectRefID = mCurrentFrame.mnId;
        mLastDirectFrame = FrameDirect(mCurrentFrame);
        mLastDirectFrame.m_pyrImg = mCurrentFrame.m_pyrImg;
        mCurrentFrame.computeImagePyramids(mImGray);
        trackPrecompute(mCurrentFrame, m_directTrackCache);
    }

    bool Tracking::NeedNewDirectRef()
    {
        if (mpPrevDirectRefID + mMaxFramesDirect < mCurrentFrame.mnId)
        {
            updateDirectReference();
            return true;
        }
        return false;
    }

    void Tracking::compareDirectVsIndirect()
    {
        std::set<MapPoint *> directSet(mvpLocalDirectInliers.begin(), mvpLocalDirectInliers.end());

        std::set<MapPoint *> indirectSet;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
                indirectSet.insert(mCurrentFrame.mvpMapPoints[i]);
        }

        int both = 0, directOnly = 0, indirectOnly = 0;

        for (MapPoint *mp: directSet)
            if (indirectSet.count(mp)) both++;
            else directOnly++;

        for (MapPoint *mp: indirectSet)
            if (!directSet.count(mp)) indirectOnly++;

        Logger<std::string>::LogInfoIII(
            "Direct vs Indirect: both=" + std::to_string(both) +
            " directOnly=" + std::to_string(directOnly) +
            " indirectOnly=" + std::to_string(indirectOnly));
    }

    void Tracking::StereoInitialization()
    {
        if (mCurrentFrame.N > 500)
        {
            // Set Frame pose to the origin
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4,CV_32F));

            // Create KeyFrame
            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

            // Insert KeyFrame in the map
            mpMap->AddKeyFrame(pKFini);

            // Create MapPoints and asscoiate to KeyFrame
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                    pNewMP->AddObservation(pKFini, i);
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                }
            }

            cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

            mpLocalMapper->InsertKeyFrame(pKFini);

            mLastFrame = Frame(mCurrentFrame);
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;

            mvpLocalKeyFrames.push_back(pKFini);
            mvpLocalMapPoints = mpMap->GetAllMapPoints();
            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

            mpMap->mvpKeyFrameOrigins.push_back(pKFini);

            //mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            mState = OK;
        }


    }

    void Tracking::MonocularInitialization()
    {
        if (!mpInitializer)
        {
            // Set Reference Frame
            if (mCurrentFrame.mvKeys.size() > 100)
            {
                mInitialFrame = Frame(mCurrentFrame);
                mLastFrame = Frame(mCurrentFrame);
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

                if (mpInitializer)
                    delete mpInitializer;

                mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

                return;
            }
        } else
        {
            // Try to initialize
            if ((int) mCurrentFrame.mvKeys.size() <= 100)
            {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }

            // Find correspondences
            ORBmatcher matcher(0.9, true);
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches,
                                                           100);

            // Check if there are enough correspondences
            if (nmatches < 100)
            {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                return;
            }

            cv::Mat Rcw; // Current Camera Rotation
            cv::Mat tcw; // Current Camera Translation
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

            if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
            {
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
                {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                    {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }

                // Set Frame Poses
                mInitialFrame.SetPose(cv::Mat::eye(4, 4,CV_32F));
                cv::Mat Tcw = cv::Mat::eye(4, 4,CV_32F);
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));
                mCurrentFrame.SetPose(Tcw);

                CreateInitialMapMonocular();
            }
        }
    }

    void Tracking::CreateInitialMapMonocular()
    {
        // Create KeyFrames
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);


        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // Insert KFs in the map
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);


        //Get scene bounds
        float maxX = std::numeric_limits<float>::min();
        float maxY = std::numeric_limits<float>::min();
        float maxZ = std::numeric_limits<float>::min();

        float minX = std::numeric_limits<float>::max();
        float minY = std::numeric_limits<float>::max();
        float minZ = std::numeric_limits<float>::max();

        // Create MapPoints and asscoiate to keyframes
        for (size_t i = 0; i < mvIniMatches.size(); i++)
        {
            if (mvIniMatches[i] < 0)
                continue;

            //Create MapPoint.
            cv::Mat worldPos(mvIniP3D[i]);

            //measure scene bounds
            maxX = std::max(maxX, mvIniP3D[i].x);
            maxY = std::max(maxY, mvIniP3D[i].y);
            maxZ = std::max(maxZ, mvIniP3D[i].z);

            minX = std::min(minX, mvIniP3D[i].x);
            minY = std::min(minY, mvIniP3D[i].y);
            minZ = std::min(minZ, mvIniP3D[i].z);



            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();

            //Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            //Add to Map
            mpMap->AddMapPoint(pMP);
        }

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // Bundle Adjustment
        Logger<std::string>::LogInfoII(
            "New Map created with " + std::to_string(mpMap->MapPointsInMap()) + " points. Frames: " +
            std::to_string(mCurrentFrame.mnId) + " - " + std::to_string(mInitialFrame.mnId));

        Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

        // Set median depth to 1
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
        {
            cout << "Wrong initialization, reseting..." << endl;
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
        {
            if (vpAllMapPoints[iMP])
            {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        float rangeX = (maxX - minX) * invMedianDepth;
        float rangeY = (maxY - minY) * invMedianDepth;
        float rangeZ = (maxZ - minZ) * invMedianDepth;

        float maxRange = std::max({rangeX, rangeY, rangeZ});
        float sceneTargetSize = 50.0f;

        mpViewer->setScaleFactor(sceneTargetSize/maxRange);

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);


        //mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);


        mpMap->NotifyMapPointsUpdated();
        mpMap->NotifyFramesUpdated();

        //Initialize direct tracking

        mCurrentDirectFrame = FrameDirect(mCurrentFrame);
        updateDirectReference();


        mState = OK;
    }

    void Tracking::CheckReplacedInLastFrame()
    {
        for (int i = 0; i < mLastFrame.N; i++)
        {
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];

            if (pMP)
            {
                MapPoint *pRep = pMP->GetReplaced();
                if (pRep)
                {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }
    }

    bool Tracking::TrackReferenceKeyFrame()
    {
        Logger<std::string>::LogInfoII(
            "Indirect Tracker: Tracking with Ref Frames: " + std::to_string(mCurrentFrame.mnId) + " - " +
            std::to_string(mpReferenceKF->mnFrameId));

        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        vector<MapPoint *> vpMapPointMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

        if (nmatches < 15)
            return false;

        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.SetPose(mLastFrame.mTcw);

        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        Logger<std::string>::LogInfoII("Indirect Tracker: Tracked Matches " + std::to_string(nmatches));

        return nmatchesMap >= 10;
    }

    void Tracking::UpdateLastFrame()
    {
        // Update pose according to reference keyframe
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back();

        mLastFrame.SetPose(Tlr * pRef->GetPose());

        if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
            return;

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);
        for (int i = 0; i < mLastFrame.N; i++)
        {
            float z = mLastFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (vDepthIdx.empty())
            return;

        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++)
        {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1)
            {
                bCreateNew = true;
            }

            if (bCreateNew)
            {
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

                mLastFrame.mvpMapPoints[i] = pNewMP;

                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            } else
            {
                nPoints++;
            }

            if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                break;
        }
    }

    bool Tracking::TrackWithMotionModel()
    {
        Logger<std::string>::LogInfoII(
            "Indirect Tracker: Tracking with motion model Frames: " + std::to_string(mCurrentFrame.mnId) + " - " +
            std::to_string(mLastFrame.mnId));

        ORBmatcher matcher(0.9, true);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();

        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

        // Project points seen in previous frame
        int th;
        if (mSensor != System::STEREO)
            th = 15;
        else
            th = 7;
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

        // If few matches, uses a wider window search
        if (nmatches < 20)
        {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);
        }

        if (nmatches < 20)
            return false;

        // Optimize frame pose with all matches
        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        //std::cout << "Matches after PNP: " << nmatchesMap << std::endl;
        if (mbOnlyTracking)
        {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

        Logger<std::string>::LogInfoII("Indirect Tracker: Tracked Matches " + std::to_string(nmatches));

        return nmatchesMap >= 10;
    }

    bool Tracking::TrackLocalMap()
    {
        // We have an estimation of the camera pose and some map points tracked in the frame.
        // We retrieve the local map and try to find matches to points in the local map.

        //In other words, search for more map points from all keyframes that observe
        //this frame's map points and also their neighbors.

        UpdateLocalMap();

        SearchLocalPoints();

        // Optimize Pose
        Optimizer::PoseOptimization(&mCurrentFrame);
        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (!mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if (!mbOnlyTracking)
                    {
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        mnMatchesInliers++;
                } else if (mSensor == System::STEREO)
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
        }

        mpMap->AddTweenFrame(mCurrentFrame);
        mpMap->NotifyFramesUpdated();

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
            return false;

        if (mnMatchesInliers < 30)
            return false;
        else
            return true;
    }

    bool Tracking::NeedNewKeyFrame()
    {
        if (mbOnlyTracking)
            return false;

        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
            return false;

        const int nKFs = mpMap->KeyFramesInMap();

        // Do not insert keyframes if not enough frames have passed from last relocalisation
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
            return false;

        // Tracked MapPoints in the reference keyframe
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

        // Local Mapping accept keyframes?
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

        // Check how many "close" points are being tracked and how many could be potentially created.
        int nNonTrackedClose = 0;
        int nTrackedClose = 0;
        if (mSensor != System::MONOCULAR)
        {
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
                {
                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                        nTrackedClose++;
                    else
                        nNonTrackedClose++;
                }
            }
        }

        bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

        // Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        if (mSensor == System::MONOCULAR)
            thRefRatio = 0.9f;

        // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
        // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
        //Condition 1c: tracking is weak
        const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

        std::string c1aString = (c1a) ? "true" : "false";
        std::string c1bString = (c1b) ? "true" : "false";
        std::string c1cString = (c1c) ? "true" : "false";
        std::string c2String = (c2) ? "true" : "false";


        if ((c1a || c1b || c1c) && c2)
        {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            if (bLocalMappingIdle)
            {
                Logger<std::string>::LogInfoI("New KF: " + std::to_string(mCurrentFrame.mnId) +
                                              ", More than max frames=" + c1aString +
                                              ", More than min frames=" + c1bString +
                                              ", Tracking is weak=" + c1cString +
                                              ", Too few tracked points=" + c2String);


                return true;
            } else
            {
                mpLocalMapper->InterruptBA();
                if (mSensor != System::MONOCULAR)
                {
                    if (mpLocalMapper->KeyframesInQueue() < 3)
                    {
                        Logger<std::string>::LogInfoI("New KF: " + std::to_string(mCurrentFrame.mnId) +
                                                      ", More than max frames=" + c1aString +
                                                      ", More than min frames=" + c1bString +
                                                      ", Tracking is weak=" + c1cString +
                                                      ", Too few tracked points=" + c2String);
                        return true;
                    } else
                        return false;
                } else
                    return false;
            }
        } else
            return false;
    }

    void Tracking::CreateNewKeyFrame()
    {
        if (!mpLocalMapper->SetNotStop(true))
            return;

        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        if (mSensor != System::MONOCULAR)
        {
            mCurrentFrame.UpdatePoseMatrices();

            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.
            vector<pair<float, int> > vDepthIdx;
            vDepthIdx.reserve(mCurrentFrame.N);
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0)
                {
                    vDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vDepthIdx.empty())
            {
                sort(vDepthIdx.begin(), vDepthIdx.end());

                int nPoints = 0;
                for (size_t j = 0; j < vDepthIdx.size(); j++)
                {
                    int i = vDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1)
                    {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }

                    if (bCreateNew)
                    {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                        MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                        pNewMP->AddObservation(pKF, i);
                        pKF->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpMap->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        nPoints++;
                    } else
                    {
                        nPoints++;
                    }

                    if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                        break;
                }
            }
        }

        mpLocalMapper->InsertKeyFrame(pKF);

        mpLocalMapper->SetNotStop(false);

        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;

        mpMap->NotifyFramesUpdated();
    }

    void Tracking::SearchLocalPoints()
    {
        // Do not search map points already matched
        for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.
                             end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP)
            {
                if (pMP->isBad())
                {
                    *vit = static_cast<MapPoint *>(NULL);
                } else
                {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

        // Project points in frame and check its visibility
        for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend;
             vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pMP->isBad())
                continue;
            // Project (this fills MapPoint variables for matching)
            if (mCurrentFrame.isInFrustum(pMP, 0.5))
            {
                pMP->IncreaseVisible();
                nToMatch++;
            }
        }

        if (nToMatch > 0)
        {
            ORBmatcher matcher(0.8);
            int th = 1;
            if (mSensor == System::RGBD)
                th = 3;
            // If the camera has been relocalised recently, perform a coarser search
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;
            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
        }
    }

    void Tracking::UpdateLocalMap()
    {
        // This is for visualization
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->NotifyMapPointsUpdated();

        // Update
        UpdateLocalKeyFrames();
        UpdateLocalPoints();
    }

    void Tracking::UpdateLocalPoints()
    {
        mvpLocalMapPoints.clear();

        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++)
        {
            KeyFrame *pKF = *itKF;
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP
                 ++)
            {
                MapPoint *pMP = *itMP;
                if (!pMP)
                    continue;
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->isBad())
                {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }

    void Tracking::UpdateLocalKeyFrames()
    {
        // Each map point vote for the keyframes in which it has been observed
        map<KeyFrame *, int> keyframeCounter;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP->isBad())
                {
                    const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end();
                         it != itend; it++)
                        keyframeCounter[it->first]++;
                } else
                {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }

        if (keyframeCounter.empty())
            return;

        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
             it != itEnd; it++)
        {
            KeyFrame *pKF = it->first;

            if (pKF->isBad())
                continue;

            if (it->second > max)
            {
                max = it->second;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.push_back(it->first);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }


        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++)
        {
            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80)
                break;

            KeyFrame *pKF = *itKF;

            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

            for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
                 itNeighKF != itEndNeighKF; itNeighKF++)
            {
                KeyFrame *pNeighKF = *itNeighKF;
                if (!pNeighKF->isBad())
                {
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                    {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            const set<KeyFrame *> spChilds = pKF->GetChilds();
            for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
            {
                KeyFrame *pChildKF = *sit;
                if (!pChildKF->isBad())
                {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                    {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            KeyFrame *pParent = pKF->GetParent();
            if (pParent)
            {
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        if (pKFmax)
        {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    bool Tracking::Relocalization()
    {
        // Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();

        // Relocalization is performed when tracking is lost
        // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

        if (vpCandidateKFs.empty())
            return false;

        const int nKFs = vpCandidateKFs.size();

        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);

        vector<PnPsolver *> vpPnPsolvers;
        vpPnPsolvers.resize(nKFs);

        vector<vector<MapPoint *> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        int nCandidates = 0;

        for (int i = 0; i < nKFs; i++)
        {
            KeyFrame *pKF = vpCandidateKFs[i];
            if (pKF->isBad())
                vbDiscarded[i] = true;
            else
            {
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                if (nmatches < 15)
                {
                    vbDiscarded[i] = true;
                    continue;
                } else
                {
                    PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                    vpPnPsolvers[i] = pSolver;
                    nCandidates++;
                }
            }
        }

        // Alternatively perform some iterations of P4P RANSAC
        // Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);

        while (nCandidates > 0 && !bMatch)
        {
            for (int i = 0; i < nKFs; i++)
            {
                if (vbDiscarded[i])
                    continue;

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                PnPsolver *pSolver = vpPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore)
                {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                if (!Tcw.empty())
                {
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint *> sFound;

                    const int np = vbInliers.size();

                    for (int j = 0; j < np; j++)
                    {
                        if (vbInliers[j])
                        {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        } else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }

                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    if (nGood < 10)
                        continue;

                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                    // If few inliers, search by projection in a coarse window and optimize again
                    if (nGood < 50)
                    {
                        int nadditional = matcher2.
                                SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                        if (nadditional + nGood >= 50)
                        {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > 30 && nGood < 50)
                            {
                                sFound.clear();
                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchByProjection(
                                    mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                                // Final optimization
                                if (nGood + nadditional >= 50)
                                {
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                }
                            }
                        }
                    }


                    // If the pose is supported by enough inliers stop ransacs and continue
                    if (nGood >= 50)
                    {
                        bMatch = true;
                        break;
                    }
                }
            }
        }

        if (!bMatch)
        {
            return false;
        } else
        {
            mnLastRelocFrameId = mCurrentFrame.mnId;
            return true;
        }
    }

    void Tracking::Reset()
    {
        cout << "System Reseting" << endl;
        if (mpViewer)
        {
            mpViewer->exit();
        }

        // Reset Local Mapping
        cout << "Reseting Local Mapper...";
        mpLocalMapper->RequestReset();
        cout << " done" << endl;

        // Reset Loop Closing
        cout << "Reseting Loop Closing...";
        mpLoopClosing->RequestReset();
        cout << " done" << endl;

        // Clear BoW Database
        cout << "Reseting Database...";
        mpKeyFrameDB->clear();
        cout << " done" << endl;

        // Clear Map (this erase MapPoints and KeyFrames)
        mpMap->clear();

        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;

        if (mpInitializer)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
        }

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();

        //if(mpViewer)
        //    mpViewer->Release();
    }

    void Tracking::ChangeCalibration(const string &strSettingPath)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3,CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1,CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        Frame::mbInitialComputations = true;
    }

    void Tracking::InformOnlyTracking(const bool &flag)
    {
        mbOnlyTracking = flag;
    }
} //namespace ORB_SLAM
