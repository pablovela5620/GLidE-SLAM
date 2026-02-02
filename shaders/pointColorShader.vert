#version 320 es
precision lowp float;

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexColor;

uniform mat4 mvpMatrix;
uniform float pointSize;
out vec3 vColor;
void main()
{
    gl_PointSize = pointSize;
    gl_Position = mvpMatrix*vec4(vertexPosition,1.0);
    vColor = vertexColor;
}