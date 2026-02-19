#version 310 es

precision mediump float;
in vec2 TexCoord;
uniform sampler2D TexSampler;
out vec4 fragColor;

void main()
{
    fragColor = texture(TexSampler, TexCoord);
    //fragColor = vec4(1.0,0.0,0.0,1.0);
}