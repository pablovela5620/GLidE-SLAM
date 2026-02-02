#version 320 es
precision lowp float;
layout (location = 0) in vec3 vertexPosition;
uniform vec3 vRGB;
out vec3 vColor;

void main()
{
    gl_PointSize = 3.0;
    gl_Position = vec4(vertexPosition,1.0);
    vColor = vRGB;
}