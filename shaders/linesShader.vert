#version 320 es
precision lowp float;

layout (location = 0) in vec3 vertexPosition;

uniform vec3 vRGB;
uniform float pointSize;
out vec3 vColor;

void main()
{
    gl_PointSize = pointSize;
    gl_Position = vec4(vertexPosition,1.0);
    vColor = vRGB;
}