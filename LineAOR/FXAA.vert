#version 420 core

layout (location=0) in vec2 aPosition;
layout (location=1)  in vec2 aTexCoord;

varying vec2 vTexCoord;

void main(void)
{
    vTexCoord = aTexCoord;    
    gl_Position = vec4(aPosition.x, aPosition.y, 0.0, 1.0);
}
