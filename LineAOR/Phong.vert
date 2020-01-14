#version 460
uniform mat4 mvp_matrix; 
uniform mat4 proj_matrix;
uniform int shader_type;
uniform float z_filter;
uniform int is_rendering_background;
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 in_tangent;
layout (location = 2) in float in_alpha;
layout (location = 3) in vec3 in_color;

layout (location = 0) out vec3 o_tangent;
layout (location = 1) out vec3 o_pos;
layout (location = 2) out float o_alpha;
layout (location = 3) out vec3 o_color;


void main()
{
    if(is_rendering_background == 0)
    	gl_Position = mvp_matrix*vec4(position,1.0f);
    else
    	gl_Position = proj_matrix*vec4(position,1.0f);
   // N=nor;//(model_matrix*vec4(nor,0.f)).xyz;
	//vec4 vertPos4 = (model_matrix*vec4(position,1.f));//eyepos
    //view = vertPos4.xyz / vertPos4.w;
    //return;
    if(gl_Position.z < z_filter)
    {
        if(shader_type == 0)
        {
            //float tmp = length(in_tangent);
            o_tangent = in_tangent;//vec3(tmp, tmp, tmp);//in_tangent;//(mv_matrix*vec4(in_tangent,0.f)).xyz;
            o_pos = position;//-(mv_matrix*vec4(position,1.f)).xyz;
            o_color = in_color;
            o_alpha = in_alpha;
            //lightv = lightpos;//lightpos + view;//(model_matrix * vec4(lightpos, 1.f)).xyz;
        }
    } else {
        gl_Position = vec4(0,0,0,1.f);
    }

    //iru = 1;
}
