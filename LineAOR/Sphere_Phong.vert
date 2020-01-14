#version 460
layout (location=0) in vec3 in_pos;
layout (location=1) in int in_id;
uniform mat4 mvp_matrix;

out vec3 o_pos;
flat out int o_id;

void main(){

    gl_Position = mvp_matrix * vec4(in_pos, 1.f);
    o_pos = in_pos;
    o_id = in_id;

}