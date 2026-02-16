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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;
class FrameDirect;

class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);

    void AddTweenFrame(Frame& pKF);
    //TODO: Rename/Remove CPU/GPU, only for testing
    void AddDirectTweenFrameCPU(FrameDirect& pKF);
    void AddDirectTweenFrameGPU(FrameDirect& pKF);
    void ClearTweenFrames();

    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<KeyFrame*> GetAllKeyFrames();

    const std::vector<Frame>& GetTweenFrames();
    const std::vector<FrameDirect>& GetDirectTweenFrames();

    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();

    uint32_t MapPointsInMap();
    uint32_t  KeyFramesInMap();

    uint32_t GetMaxKFid();

    void clear();

    vector<KeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;


    void NotifyMapPointsUpdated(){ maMapPointUpdateNumber.fetch_add(1); }
    void NotifyFramesUpdated(){ maFramesUpdateNumber.fetch_add(1); }
    uint32_t GetMapPointsUpdateNumber() const { return maMapPointUpdateNumber.load(); }
    uint32_t GetFramesUpdateNumber() const { return maFramesUpdateNumber.load(); }
protected:
    std::set<MapPoint*> mspMapPoints;
    std::set<KeyFrame*> mspKeyFrames;

    std::vector<Frame> mspTweenFrames;

    //TODO: Rename/Remove CPU/GPU, only for testing
    std::vector<FrameDirect> mspTweenDirectFramesCPU;
    std::vector<FrameDirect> mspTweenDirectFramesGPU;

    std::vector<MapPoint*> mvpReferenceMapPoints;

    uint32_t mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;

private:
    std::atomic<uint32_t> maMapPointUpdateNumber{0};
    std::atomic<uint32_t> maFramesUpdateNumber{0};

};

} //namespace ORB_SLAM

#endif // MAP_H
