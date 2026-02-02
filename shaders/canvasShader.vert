#version 320 es

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec2 textureCoord;

out vec2 TexCoord;

void main()
{
	gl_Position = vec4(vertexPosition,1.0);
    TexCoord = textureCoord;
}