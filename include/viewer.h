#ifndef VIEWER_H
#define VIEWER_H

//system
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <memory>
#include <thread>
#include <chrono>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <map>
#include <cmath>
#include <algorithm>

//other
#include <glad/glad.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>

// Undefine 'Success' macro if defined
#ifdef Success
#undef Success
#endif
#ifdef Status
#undef Status
#endif


//OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

//maths library
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_access.hpp>

//own
#include <oneapi/tbb/profiling.h>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "System.h"
#include "SlamSettings.h"
#include "Map.h"
#include "Logger.h"

namespace ORB_SLAM2
{
    class System;
}

static const char *eglGetErrorString(EGLint error)
{
    switch (error)
    {
        case EGL_SUCCESS:
            return "EGL_SUCCESS: No error";
        case EGL_NOT_INITIALIZED:
            return "EGL_NOT_INITIALIZED: EGL is not initialized";
        case EGL_BAD_ACCESS:
            return "EGL_BAD_ACCESS: EGL cannot access a requested resource";
        case EGL_BAD_ALLOC:
            return "EGL_BAD_ALLOC: EGL failed to allocate resources";
        case EGL_BAD_ATTRIBUTE:
            return "EGL_BAD_ATTRIBUTE: Unrecognized attribute or attribute value";
        case EGL_BAD_CONTEXT:
            return "EGL_BAD_CONTEXT: An EGLContext argument does not name a valid EGL rendering context";
        case EGL_BAD_CONFIG:
            return "EGL_BAD_CONFIG: An EGLConfig argument does not name a valid EGL frame buffer configuration";
        case EGL_BAD_DISPLAY:
            return "EGL_BAD_DISPLAY: An EGLDisplay argument does not name a valid EGL display connection";
        case EGL_BAD_SURFACE:
            return "EGL_BAD_SURFACE: An EGLSurface argument does not name a valid EGL drawing surface";
        case EGL_BAD_MATCH:
            return
                    "EGL_BAD_MATCH: Arguments are inconsistent (for example, a context requires buffers not supplied by a surface)";
        case EGL_BAD_PARAMETER:
            return "EGL_BAD_PARAMETER: One or more argument values are invalid";
        case EGL_BAD_NATIVE_PIXMAP:
            return "EGL_BAD_NATIVE_PIXMAP: An EGLNativePixmapType argument does not refer to a valid native pixmap";
        case EGL_BAD_NATIVE_WINDOW:
            return "EGL_BAD_NATIVE_WINDOW: An EGLNativeWindowType argument does not refer to a valid native window";
        case EGL_CONTEXT_LOST:
            return "EGL_CONTEXT_LOST: A power management event has occurred";
        default:
            return "Unknown EGL error code";
    }
}

enum class EventTypes
{
    Empty = 0,
    WindowClose,
    WindowResize,
    WindowFocus,
    WindowLostFocus,
    WindowMoved,
    AppTick,
    AppUpdate,
    AppRender,
    KeyPressed,
    KeyReleased,
    MousePressed,
    MouseReleased,
    MouseMoved,
    MouseScrolled,
    GUIUpdate,
    MapUpdate
};

namespace UIEvents
{
    class UIEvent
    {
    public:
        UIEvent(const std::string &name, const EventTypes &event) : m_name(name), m_eventType(event)
        {
        }

        virtual void setName(const std::string &name) { m_name = name; }
        virtual void setType(const EventTypes &type) { m_eventType = type; }
        virtual const std::string &getName() { return m_name; }
        virtual const EventTypes &getType() const { return m_eventType; }

        virtual const std::string toString() const = 0;

    protected:
        std::string m_name;
        EventTypes m_eventType;
    };

    class MouseButtonPressedUIEvent : public UIEvent
    {
    public:
        MouseButtonPressedUIEvent(const int button, const bool action) :
       m_button(button), m_action(action), UIEvent("MouseButtonPressed", EventTypes::MousePressed)
        {
        }

        int getButton() const {return m_button;}
        bool getAction() const {return m_action;}
        const std::string toString() const override
        {
            std::stringstream ss;
            ss << m_name << " button: " << std::to_string(m_button) << " action: " << std::to_string(m_action);
            return ss.str();
        }

    private:
        int m_button;
        bool m_action;
    };

    class MouseButtonReleasedUIEvent : public UIEvent
    {
    public:
        MouseButtonReleasedUIEvent(const int button, const bool action) :
       m_button(button), m_action(action), UIEvent("MouseButtonPressed", EventTypes::MouseReleased)
        {
        }

        int getButton() const {return m_button;}
        bool getAction() const {return m_action;}
        const std::string toString() const override
        {
            std::stringstream ss;
            ss << m_name << " button: " << std::to_string(m_button) << " action: " << std::to_string(m_action);
            return ss.str();
        }

    private:
        int m_button;
        bool m_action;
    };

    class MouseWheelUIEvent : public UIEvent
    {
        public:
            MouseWheelUIEvent(const int wheel) : m_wheel(wheel), UIEvent("MouseWheel", EventTypes::MouseScrolled){}

        int getWheel() const {return m_wheel;}
        const std::string toString() const override
            {
                std::stringstream ss;
                ss << m_name << " wheel: " << std::to_string(m_wheel);
                return ss.str();
            }
        private:
        int m_wheel;
    };

    class MouseMovedUIEvent : public UIEvent
    {
    public:
        MouseMovedUIEvent(const float x, const float y) : m_x(x), m_y(y), UIEvent("MouseMoved", EventTypes::MouseMoved)
        {
        }

        float getX() const { return m_x; }
        float getY() const { return m_y; }
        const std::string toString() const override
        {
            std::stringstream ss;
            ss << m_name << " x: " << m_x << " y: " << m_y;
            return ss.str();
        }

    private:
        float m_x;
        float m_y;
    };

    class WindowResizeUIEvent : public UIEvent
    {
    public:
        WindowResizeUIEvent(const float x, const float y) : m_width(x), m_height(y),
                                                            UIEvent("WindowResize", EventTypes::WindowResize)
        {
        }

        const std::string toString() const override
        {
            std::stringstream ss;
            ss << m_name << " width: " << m_width << " height: " << m_height;
            return ss.str();
        }

    private:
        float m_width;
        float m_height;
    };

    class WindowCloseUIEvent : public UIEvent
    {
    public:
        WindowCloseUIEvent() : UIEvent("WindowClose", EventTypes::WindowClose)
        {
        }

        const std::string toString() const override
        {
            std::stringstream ss;
            ss << m_name;
            return ss.str();
        }
    };

    class KeyPressUIEvent : public UIEvent
    {
    public:
        KeyPressUIEvent(const int key, const int action) : m_key(key), m_action(action),
                                                           UIEvent("KeyPress", EventTypes::KeyPressed)
        {
        }

        const int getKey() const { return m_key; }
        const int getAction() const { return m_action; }

        const std::string toString() const override
        {
            std::stringstream ss;
            ss << m_name << " key: " << m_key;
            return ss.str();
        }

    private:
        int m_key;
        int m_action;
    };

    class KeyReleaseUIEvent : public UIEvent
    {
    public:
        KeyReleaseUIEvent(const int key, const int action) : m_key(key), m_action(action),
                                                             UIEvent("KeyRelease", EventTypes::KeyReleased)
        {
        }

        const int getKey() const { return m_key; }
        const int getAction() const { return m_action; }

        const std::string toString() const override
        {
            std::stringstream ss;
            ss << m_name << " key: " << m_key;
            return ss.str();
        }

    private:
        int m_key;
        int m_action;
    };

    class UIEventManager
    {
    public:
        void subscribe(const EventTypes &event, std::function<void(const UIEvent &)> callback)
        {
            m_subscribers[event] = callback;
        }

        void post(const UIEvent &event)
        {
            EventTypes eventType = event.getType();
            if (m_subscribers.find(eventType) == m_subscribers.end())
                return;
            auto func = m_subscribers.find(eventType);
            func->second(event);
        }

    private:
        std::map<EventTypes, std::function<void(const UIEvent &)> > m_subscribers;
    };
};

class ViewerUtil final
{
public:
    // Conversion utilities for OpenGL
    static void convertToGL(const std::vector<glm::vec3> &points, std::vector<GLfloat> &glPoints);
    static void convertToGL(const std::vector<glm::vec2> &points, std::vector<GLfloat> &glPoints);

    // Translation and Orientation Conversion
    inline static glm::vec3 convertTranslation(const glm::vec3 &t, const glm::vec3 &v)
    {
        glm::mat3 M(0.0f);
        M[0][std::abs(v[0]) - 1] = (v[0] < 0) ? -1 : 1;
        M[1][std::abs(v[1]) - 1] = (v[1] < 0) ? -1 : 1;
        M[2][std::abs(v[2]) - 1] = (v[2] < 0) ? -1 : 1;
        return M * t;
    }

    inline static glm::mat3 convertOrientation(const glm::mat3 &R, const glm::vec3 &v)
    {
        glm::mat3 M(0.0f);
        M[0][std::abs(v[0]) - 1] = (v[0] < 0) ? -1 : 1;
        M[1][std::abs(v[1]) - 1] = (v[1] < 0) ? -1 : 1;
        M[2][std::abs(v[2]) - 1] = (v[2] < 0) ? -1 : 1;
        return R * glm::transpose(M);
    }

    struct Quaternion
    {
        float x, y, z, w;

        Quaternion(float _x, float _y, float _z, float _w)
            : x(_x), y(_y), z(_z), w(_w) {}

        // Conjugate of the quaternion
        static Quaternion conjugate(float _x, float _y, float _z, float _w)
        {
            return Quaternion(-_x, -_y, -_z, _w);
        }

        // Normalize the quaternion
        Quaternion normalize() const
        {
            float mag = std::sqrt(w * w + x * x + y * y + z * z);
            return Quaternion(x / mag, y / mag, z / mag, w / mag);
        }

        // Quaternion multiplication (Quaternion * Quaternion)
        Quaternion operator*(const Quaternion &r) const
        {
            return Quaternion(
                (w * r.x) + (x * r.w) + (y * r.z) - (z * r.y),
                (w * r.y) + (y * r.w) + (z * r.x) - (x * r.z),
                (w * r.z) + (z * r.w) + (x * r.y) - (y * r.x),
                (w * r.w) - (x * r.x) - (y * r.y) - (z * r.z)
            );
        }

        // Quaternion-vector multiplication (Quaternion * glm::vec3)
        Quaternion operator*(const glm::vec3 &v) const
        {
            return Quaternion(
                (w * v.x) + (y * v.z) - (z * v.y),
                (w * v.y) + (z * v.x) - (x * v.z),
                (w * v.z) + (x * v.y) - (y * v.x),
                -(x * v.x) - (y * v.y) - (z * v.z)
            );
        }
    };

    // Function to rotate a vector by an angle around an axis using quaternions
    inline static glm::vec3 rotateAngleAxis(const glm::vec3 &vector, float angle, const glm::vec3 &axis)
    {
        float halfSinAngle = std::sin(angle / 2.0f);
        Quaternion rotationQ(axis.x * halfSinAngle, axis.y * halfSinAngle, axis.z * halfSinAngle, std::cos(angle / 2.0f));
        Quaternion conjugateQ = Quaternion::conjugate(rotationQ.x, rotationQ.y, rotationQ.z, rotationQ.w);

        Quaternion result = rotationQ * vector * conjugateQ;
        return glm::vec3(result.x, result.y, result.z);
    }

    // Smooth step function
    inline static float smoothStep(float x, float l0, float l1)
    {
        x = (x < l0) ? l0 : x;
        x = (x > l1) ? l1 : x;
        x = (x - l0) / (l1 - l0);  // Normalize x to range [0, 1]
        return x * x * (3 - 2 * x); // Apply smooth step formula
    }
    inline static float getFPS(std::deque<float>& timeFrames, float dt, const int N)
    {
        timeFrames.push_back(dt);
        if(timeFrames.size() > N) timeFrames.pop_front();

        float totalFramesTime = 0.0f;
        for(size_t i = 0; i < timeFrames.size(); i++)
            totalFramesTime += timeFrames[i];
        return ((float)timeFrames.size() / totalFramesTime);
    }

private:
    // Prevent instantiation and copying
    ViewerUtil() = delete;
    ViewerUtil &operator=(const ViewerUtil &) = delete;
    ViewerUtil(const ViewerUtil &) = delete;
};

class Shader
{
public:
    Shader()
    {
    }

    ~Shader()
    {
    }

    //not allowed copies
    Shader(const Shader &) = delete;

    Shader &operator =(const Shader &) = delete;

    void use()
    {
        if (m_shaderProgram > 0 && m_isLinked)
            glUseProgram(m_shaderProgram);
    }

    bool link();

    bool compile(GLenum shaderType, const std::string &shaderFile);

    int getHandle() const { return m_shaderProgram; }

    bool setHandle(GLuint handle);

    bool isLinked() const { return m_isLinked; }

    void setUniform(const char *name, const glm::mat4 &m);

    void setUniform(const char *name, const glm::vec2 &v);

    void setUniform(const char *name, const glm::vec3 &v);

    void setUniform(const char *name, float val);

    void setUniform(const char *name, int val);

private:
    int getUniformLocation(const char *name);

    void detachAndDeleteShaders();

    static std::string readFile(const std::string &path);

    void findUniformLocations();

    void findAttributeLocations();

private:
    GLuint m_shaderProgram;
    std::vector<GLuint> m_compiledShaders;
    std::map<std::string, GLuint> m_uniformLocations;
    std::map<std::string, GLuint> m_attributeLocations;
    bool m_isLinked{false};
};

class Canvas
{
public:
    Canvas(const int width, const int height) : m_width(width), m_height(height) { initialize(); }
    ~Canvas(){deleteBuffers();}
    void update();

    void updateImage(const cv::Mat &image);

    void render() const;

private:
    void initialize();

    void bindTexture();

    void initializeBuffers(std::vector<GLfloat> *glPoints, std::vector<GLuint> *indices,
                           std::vector<GLfloat> *texCoords);

    void deleteBuffers();

private:
    GLuint m_vao;
    GLuint m_vbo;
    std::vector<GLuint> m_buffers;
    GLuint m_N;
    cv::Mat m_image;
    GLuint m_texture;

    int m_width{64};
    int m_height{64};
};

class GraphicPrimitive
{
public:
    GraphicPrimitive()
    {
    }

    GraphicPrimitive(const glm::mat4 &pose) : m_pose(pose)
    {
    }

    virtual ~GraphicPrimitive()
    {
    }

    virtual void render() const;

    void setTransform(const glm::mat4 &newPose)
    {
    }

    void setScale(const float v) { m_scale = v; }
    const float getScale(void) const { return m_scale; }
    const glm::mat4 &getPose(void) const { return m_pose; }

    void setPose(const glm::mat4 &pose)
    {
        m_pose = pose;
        m_t = glm::vec3(pose[3]);
    }

    const glm::vec3 &getPosition(void) const { return m_t; }

    virtual void deleteBuffers();

    virtual void initializeEmptyBuffer();

    void clear() { deleteBuffers(); }

    void loadPoints(const std::vector<glm::vec3> &points);

    GLuint getN() const { return m_N; }

protected:
    virtual void initializeBuffers(std::vector<GLfloat> *points);

    virtual void initializeBuffers(std::vector<GLfloat> *glPoints, std::vector<GLfloat> *glpointsColors);

    virtual void initializeBuffers(const std::vector<GLfloat> *points, const std::vector<GLuint> *indices);

    virtual void updateBuffer(const std::vector<GLfloat> *points);
    virtual void updateBuffer(const std::vector<GLfloat> *points,const std::vector<GLfloat> *colorPoints);

    virtual void loadPoints(const std::vector<glm::vec3> &points, const std::vector<glm::vec3> &pointsColor);

protected:
    GLsizei m_N;
    GLuint m_vao;
    GLuint m_vbo;
    std::vector<GLuint> m_buffers;
    glm::mat4 m_pose;
    glm::mat3 m_R;
    glm::vec3 m_t;

private:
    virtual void initialize();

private:
    float m_scale{1.0f};
};

class Lines2D : public GraphicPrimitive
{
public:
    //update points as pair of points for each line
    void updatePoints(const std::vector<glm::vec3> &points);

    void render() const override;

private:
    bool m_isInitialized{false};
};

class SquareGizmo : public GraphicPrimitive
{
public:
    void initialize() override;

private:
    uint32_t m_ID{0};
    char m_size{2};
};

class AxisGizmo : public GraphicPrimitive
{
public:
    AxisGizmo() { initialize(); };

private:
    void initialize() override;
};

class FrameGizmo : public GraphicPrimitive
{
public:
    FrameGizmo()
    {
    };

    FrameGizmo(const char frameType) : m_frameType(frameType)
    {
    };

    FrameGizmo(const char frameType, const glm::mat4 &pose, uint32_t id) : GraphicPrimitive(pose),
        m_frameType(frameType), m_ID(id)
    {
    };

    void renderAxisGizmos() const;

    void setParentNode(FrameGizmo *f);

    FrameGizmo *getParentNode(void) const { return m_parent; }
    void removeParentNode(void) { m_parent = nullptr; }

    void initialize() override;

private:
    char m_frameType{0};
    uint32_t m_ID{0};
    AxisGizmo m_axisGizmo;
    FrameGizmo *m_parent{nullptr};
};

class PathGizmo : public GraphicPrimitive
{
public:
    void updatePoints(const std::vector<glm::vec3> &points);

    void render() const override;

private:
    bool m_isInitialized{false};
};

class TrailGizmo : public GraphicPrimitive
{
public:
    TrailGizmo()
    {
    }

    ~TrailGizmo()
    {
    }

    void render() const override;
};

class PointCloud : public GraphicPrimitive
{
public:
    ~PointCloud() { deleteBuffers(); }

    void loadPoints(const std::vector<glm::vec3> &points);

    void updatePoints(const std::vector<glm::vec3> &points);
    void updatePoints(const std::vector<glm::vec3> &points,const std::vector<glm::vec3> &pointsColor);

    void render() const override;

    void initializeEmptyBuffer() override;
    void setColor(const glm::vec3 color) { m_color = color; }
    const glm::vec3 getColor(void) const { return m_color; }

private:
    glm::vec3 m_color{1.0f, 1.0f, 1.0f};
};

class Camera
{
public:
    Camera(float width, float height, const glm::vec3 pos, const glm::vec3 target, const glm::vec3 up)
            : m_width(width), m_height(height), m_forward(target), m_up(up)
    {
        m_position = pos;
        initialize();
    }


    void onLook(int x, int y);
    void onMove(int key, int mode);
    void onZoom(int dir) {m_zoom = dir;}
    void setTarget(const glm::mat4& t){m_target = t;}
    const glm::mat4 &getViewMatrix()
    {
        setTransform();
        return m_viewMatrix;
    }
    const glm::mat4 &getProjectionMatrix()
    {
        setTransform();
        return m_projectionMatrix;
    }
    const glm::mat4 &getViewProjectionMatrix()
    {
        setTransform();
        m_viewProjectionMatrix = m_viewMatrix * m_projectionMatrix;
        return m_viewProjectionMatrix;
    }
    void update(float t);
    bool isFollowing() const {return m_follow;}
    void setFollow(bool follow, const float distance)
    {m_follow = follow; m_followDistance = distance;}

private:
    void updateMove(float t);
    void follow();
    void setTransform();
    void setProjectionTransform();
    void setPosition(const glm::vec3 &pos) { m_position = pos; };
    void initialize();
private:
    uint8_t m_keyMap{0};
    int m_zoom{0};

    glm::vec3 m_right;
    glm::vec3 m_up;
    glm::vec3 m_forward;

    float m_horAngle{0.0f};
    float m_verAngle{0.0f};
    float m_minMaxVerAngle{-85};

    int m_mouseX{0};
    int m_mouseY{0};

    float m_dx{0};
    float m_dy{0};
    float m_sdx{0.0f};
    float m_sdy{0.0f};

    float m_horSensitivity{1.0f };
    float m_verSensitivity{1.0f };
    float m_stepSensitivity{ 0.5f };
    float m_maxVerAngle{-85.0f };

    bool m_isInitialized{false};

    bool m_follow{false};
    glm::mat4 m_target;
    float m_followDistance{1.0f};

    glm::mat4 m_viewMatrix;
    glm::mat4 m_projectionMatrix;
    glm::mat4 m_viewProjectionMatrix;

    glm::mat4 m_transform;
    glm::vec3 m_position;
    glm::mat3 m_orientation;

    int m_xPos{100};
    int m_yPos{100};
    int m_width{640};
    int m_height{480};

    float m_fov{60.0f};
    float m_ar{1.0f};
    float m_near{0.1f};
    float m_far{100.0f};

    struct
    {
        float vForward{ 0.0f };
        float vSide{ 0.0f };
        float iForward{ 0.5f };
        float iSide{ 0.5f };
        float iZoom{10.0f};
        float stopSpeed{0.05f};
        float deacceleration{ 0.8f };
    }m_motion;

};

using namespace UIEvents;
class GuiWindow
{
public:
    GuiWindow();

    GuiWindow(int x, int y, int width, int height, const std::string &title);

    GuiWindow(int x, int y, int width, int height, const std::string &title, EGLContext otherContext, EGLDisplay otherDisplay, EGLConfig otherConfig);

    ~GuiWindow();

    static GuiWindow *createWindow();

    static GuiWindow *createWindow(int x, int y, int width, int height, const std::string &title);

    static GuiWindow *createWindow(int x, int y, int width, int height, const std::string &title, EGLContext otherContext, EGLDisplay otherDisplay, EGLConfig otherConfig);

#ifdef USE_EGL_SDL
    EGLDisplay getDisplay() const { return m_eglDisplay; }
    EGLConfig getConfig() const { return m_eglConfig; }
    EGLContext getContext() const { return m_eglContext; }
    EGLSurface getSurface() const { return m_eglSurface; }
    SDL_Window *getWindow() const { return m_window; }

    bool setDisplay(EGLDisplay display) {m_eglDisplay = display;};
    bool setConfig(EGLConfig config) {m_eglConfig = config;};
#else
    GLFWwindow* Getwindow() const { return m_windowGLFW; }
#endif

    bool initializeWindow(EGLContext sharedContext);
    bool initializeWindowShared(EGLContext sharedContext, EGLDisplay sharedDisplay, EGLConfig sharedConfig);

    void setUICallBacks();

    void setEventCallback(const std::function<void(const UIEvent&)>& callback) { m_windowData.callback = callback; }

    void onUpdateWindow();

    void printVersions();

    void exit();

private:
    void cleanup();

public:
    struct WindowData
    {
        int col;
        int row;
        std::function<void(UIEvent &)> callback;
    };
private:

    WindowData m_windowData;

    int m_displayIndex;
    int m_xOffset;
    int m_yOffset;
    int m_width;
    int m_height;
    std::string m_title;

    //if using EGL+SDL
    SDL_Window *m_window{nullptr};

    EGLDisplay m_eglDisplay;
    EGLConfig m_eglConfig;
    EGLContext m_eglContext;
    EGLSurface m_eglSurface;
};

class GPUCompute
{
    public:

    GPUCompute(){};
    void initialize(int w,int h,int levels,float scaleFactor, float fx, float fy, float cx, float cy);
    bool setShaders(GLuint gHHandle, GLuint resizeHandle);
    bool buildPyramid(cv::Mat image);
    bool preCompute(const std::vector<glm::vec3>& mapPoints, const cv::Mat& pose);
    bool track(const cv::Mat& image, const cv::Mat poseIinitial,float outB[6], float& outChi2, int& outN);
    bool shutDown();

private:
    void initializeImagePyramids();

private:

    int m_width{0};
    int m_height{0};
    int m_nLevels{0};
    float m_scaleFactor{1.0f};
    float m_fx{0.0f};
    float m_fy{0.0f};
    float m_cx{0.0f};
    float m_cy{0.0f};


    std::vector<GLuint> m_pyrTexHandles;
    std::vector<GLuint> m_tempTexHandles;
    std::vector<GLuint> m_blurTexHandles;

    std::vector<int> m_levelWidth;
    std::vector<int> m_levelHeight;

    std::vector<float> m_gaussWeights;
    float m_gaussSigma{1.0f};

    //pyramid shader handles
    GLuint m_shaderGauss{0};
    GLuint m_shaderResize{0};
    GLint m_blurDirectionUniform{-1};
    GLint m_scaleFactorUniform{-1};
};

class Viewer
{
public:
    //TODO: separeate methods into private/public
    Viewer(ORB_SLAM2::System* system, SlamSettings* slamSettings) : m_system(system), m_slamViewerSettings(slamSettings){};

    bool initialize();
    void run();
    void stop();
    void exit();

    void updateFrames2D();
    void updateFrames3D();

    void updateMapPoints();
    void updateKFrames();
    void updateTweenIndirectFrames();
    void updateTweenDirectFrames();

    void updateIndirectFeatureMatches(const cv::Mat &image, const std::vector<cv::KeyPoint> &kpts1,const std::vector<cv::KeyPoint> &kpts2, const std::vector<float> &d);

    void setMap(ORB_SLAM2::Map* map) { m_map = map; }

    void setViewMatrix(const glm::mat4 view) { m_vMatrix = view; }
    void setProjectionMatrix(const glm::mat4 proj) { m_pMatrix = proj; }
    void setModelMatrix(const glm::mat4 model) { m_mMatrix = model; }

    void setSquareUpdateFlag(const char &state);

    void render();
    void renderFrames2D();
    void renderMap3D();

    bool setActiveCamera(std::shared_ptr<Camera> camera);
    void setMatrices();

    void debugGetComputeShaderTime()
    {
        if (m_computeShader_done) { std::cout << "time: " << std::to_string(m_computeShader_dtAvg) << std::endl; }
    }

    bool checkPause(){return m_pauseSimulation.load();}
    void setPause() {m_pauseSimulation.store(true);}
    void setScaleFactor(const float scale) { m_scaleFactor = scale; }

    void updateSourceImage(const cv::Mat& image);
private:
    void initializeWindows();
    void initializeProjectionMatrix();
    void initializeShaders();
    void initializeBuffers();
    void initializeMapPoints();
    void initializeCamera();

    void printVersions();
    void PollEvents();

    void shutdown();

    //events
    void onEvent(const UIEvent& e) {m_uiEventManager.post(e);}
    void onMouse(const UIEvent& e);
    void onKeyboard(const UIEvent& e);
    void onWindow(const UIEvent& e);
    void setEventCallback(std::function<void(const UIEvent &)> callback) {m_windowData.callback = callback;}
    void ensureWindowContext(EGLDisplay display, EGLSurface surface, EGLContext context);
private:

    ORB_SLAM2::System* m_system{nullptr};

    //Window
    GuiWindow *m_windowFrames2D{nullptr};
    GuiWindow *m_windowMap3D{nullptr};

    //Viewer visuals
    int m_width{640};
    int m_height{480};

    std::string m_windowFramesTitle{"No title"};
    std::string m_windowMapTitle{"No title"};

    float m_scaleFactor{1.0f};
    float m_fov{90.0f};
    float m_far{1000.0f};
    float m_near{1.0f};
    float m_featuresMaxDepth{10.0f};
    glm::mat4 m_p;
    glm::mat4 m_k;

    glm::vec3 m_currentKeyFrameColor{glm::vec3(0.0f)};
    glm::vec3 m_AllKeyFrameColor{glm::vec3(0.0f)};
    glm::vec3 m_tweenFrameDirectColor{glm::vec3(0.0f)};
    glm::vec3 m_tweenFrameColor{glm::vec3(0.0f)};
    glm::vec3 m_mapPointsColor{glm::vec3(0.0f)};
    glm::vec3 m_mapPointsRefColor{glm::vec3(0.0f)};
    glm::vec3 m_featureLinesColor{glm::vec3(0.0f)};

    EGLDisplay m_eglDisplay {EGL_NO_DISPLAY};
    EGLConfig m_eglConfig {};
    EGLContext m_eglContext {EGL_NO_CONTEXT};
    EGLSurface m_eglSurface {EGL_NO_SURFACE};

    ORB_SLAM2::Map* m_map{nullptr};

    //updates from other threads
    //use use number instead of bool since different methods/asynchronous update
    uint32_t ma_LastMapPointUpdateNumber;
    uint32_t ma_LastFramesUpdateNumber;

    //tracking window graphic elements (2D)
    std::vector<glm::vec3> m_matchedFeature2DLines;
    cv::Mat m_canvasImage;
    Lines2D* m_trackLinesGfx{nullptr};

    std::vector<glm::vec3> m_directCorner2DPoints;
    std::vector<glm::vec3> m_directHighGrad2DPoints;

    //mapping window graphic elements
    //camera frames
    FrameGizmo* m_currentKeyFrameGfx{nullptr};
    std::map<uint32_t, FrameGizmo*> m_keyFramesGfx;
    std::map<uint32_t, FrameGizmo*> m_tweenFramesDirectGfx;
    std::map<uint32_t, FrameGizmo*> m_tweenFramesGfx;

    //point clouds
    PointCloud* m_mapPointsGfx{nullptr};
    PointCloud* m_mapPointsRefGfx{nullptr};

    //canvas
    Canvas* m_canvasIndirectTracking{nullptr};
    Canvas* m_canvasDirectTracking{nullptr};

    UIEventManager m_uiEventManager;

    std::map<std::string, std::shared_ptr<Shader> > m_shaders;
    std::shared_ptr<Camera> m_activeCamera{nullptr};

    GLuint renderFBO{0};
    GLuint depthFBO{0};

    glm::mat4 m_mMatrix{glm::mat4(1.0f)};
    glm::mat4 m_vMatrix{glm::mat4(1.0f)};
    glm::mat4 m_pMatrix{glm::mat4(1.0f)};
    glm::mat4 m_mvpMatrix{glm::mat4(1.0f)};

    SlamSettings *m_slamViewerSettings{NULL};

    std::mutex mMutexUpdate;
    std::condition_variable m_cv;
    std::mutex m_viewerMutex;
    std::atomic<bool> m_stop{false};

    //for debugging
    double m_computeShader_dtAvg{0.0};
    double m_computeShader_total{0.0};
    uint32_t m_computeShader_nSamples{0};
    std::atomic<bool> m_computeShader_done{false};

    GuiWindow::WindowData m_windowData{};

    //timing variables (delta time, FPS)
    float m_oldTime{0.0f};
    float m_newTime{0.0f};
    std::deque<float> m_frameTimes;
    const uint32_t m_N = 100;


    //pause non-rendering stuff
    std::atomic<bool> m_pauseSimulation{false};

    bool m_isInitialized{false};

    //Compute Shaders (Image Processing)
    GPUCompute* m_gpuCompute{nullptr};
    std::thread                m_computeThread;
    std::mutex                 m_jobQueueMutex;
    std::condition_variable    m_jobQueueCondition;

    std::mutex                 m_publishMutex;
    std::atomic<uint64_t>      m_publishSequence{0};

};


#endif //VIEWER_H
