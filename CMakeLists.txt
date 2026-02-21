cmake_minimum_required(VERSION 3.10)
project(GLidE_SLAM)
add_definitions(-DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)
# ------------------------------------------------------------------------------
# Build type
# ------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wall -O0 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -march=native")

# ------------------------------------------------------------------------------
# C++11
# ------------------------------------------------------------------------------
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)

if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  add_definitions(-DCOMPILEDWITHC11)
  message(STATUS "Using C++11")
else()
  message(FATAL_ERROR "C++11 not supported by compiler")
endif()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# ---- OpenCV (FORCE CONFIG MODE, OpenCV 4) ----
# ---- OpenCV ----
find_package(OpenCV 4 REQUIRED)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 3 REQUIRED)
endif()
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ---- Eigen ----
find_package(Eigen3 3.1.0 REQUIRED)

# ---- Pangolin ----

option(USE_EGL_SDL "Use EGL and SDL instead of GLFW" ON)

if(USE_EGL_SDL)
  add_definitions(-DUSE_EGL_SDL)
endif()

set(OpenGL_GL_PREFERENCE "GLVND")
include_directories(${PROJECT_SOURCE_DIR}/Thirdparty/glad/include)

find_package(OpenGL REQUIRED)
if(OpenGL_FOUND)
  message(STATUS "OpenGL Found. Include dirs:" ${OpenGL_INCLUDE_DIRS})
endif()

if(USE_EGL_SDL)
  find_package(SDL2 REQUIRED)
  if(SDL2_FOUND)
    message(STATUS "SDL2 Found. Include dirs: ${SDL2_INCLUDE_DIRS}")
  else()
    message(FATAL_ERROR "SDL2 NOT Found.")
  endif()

  include_directories(/usr/include/EGL /usr/include/GLES )
  find_library(EGL_LIBRARY NAMES EGL libEGL.so PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
  if(EGL_LIBRARY)
    message(STATUS "EGL library found: ${EGL_LIBRARY}")
  else()
    message(FATAL_ERROR "EGL library NOT found.")
  endif()
else()
  find_package(glfw3 REQUIRED)
  if(GLFW_FOUND)
    message(STATUS "GLFW Found. Include dirs: ${glfw3_INCLUDE_DIRS}")
  else()
    message(FATAL_ERROR "GLFW NOT Found.")
  endif()
endif()

# ------------------------------------------------------------------------------
# Includes
# ------------------------------------------------------------------------------
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

# ------------------------------------------------------------------------------
# Library
# ------------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/Tracking.cc
        src/LocalMapping.cc
        src/LoopClosing.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/Converter.cc
        src/MapPoint.cc
        src/KeyFrame.cc
        src/Map.cc
        src/Optimizer.cc
        src/PnPsolver.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Initializer.cc
        include/ImageHandler.h
        include/GLideUtils.h
        src/ImageHandler.cc
        include/GLideEngine.h
        src/GLideEngine.cpp
        ${PROJECT_SOURCE_DIR}/Thirdparty/glad/src/glad.c
        include/GLideUtils.h
        src/GLideUtils.cpp
        include/MotionModel.h
)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

if(USE_EGL_SDL)
  include_directories(${SDL2_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES} ${EGL_LIBRARY} OpenGL::GL)
else()
  target_link_libraries(${PROJECT_NAME} glfw OpenGL::GL)
endif()

# Copy shader files
file(GLOB SHADER_FILES
        "${CMAKE_SOURCE_DIR}/shaders/*.vert"
        "${CMAKE_SOURCE_DIR}/shaders/*.frag"
        "${CMAKE_SOURCE_DIR}/shaders/*.comp"
)
file(COPY ${SHADER_FILES} DESTINATION "${CMAKE_BINARY_DIR}/shaders")


# ------------------------------------------------------------------------------
# Examples
# # ------------------------------------------------------------------------------

 set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
 add_executable(mono_tum Examples/Monocular/mono_tum.cc)
 target_link_libraries(mono_tum ${PROJECT_NAME})

 add_executable(mono_kitti Examples/Monocular/mono_kitti.cc)
 target_link_libraries(mono_kitti ${PROJECT_NAME})

 add_executable(mono_euroc Examples/Monocular/mono_euroc.cc)
 target_link_libraries(mono_euroc ${PROJECT_NAME})
