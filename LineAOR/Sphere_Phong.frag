#version 460
in vec3 o_pos;
flat in int o_id;

uniform mat4 mv_matrix;
uniform vec3 lightpos;
uniform sampler1D texSphereInfo;
uniform float texSphereInfoCounts;
out vec4 color;

float _diffuse=0.9f;
float _specular=.7f;
float _ambient=.3f;
vec4 lightcolor = vec4(1.f,1.f,1.f, 1.f);
void main(){    
    //color = vec4(float(o_id),1.f,1.f,1.f);
    //return;
    vec3 this_color = vec3(texelFetch(texSphereInfo, int(o_id)*7 + 0, 0).r,
                           texelFetch(texSphereInfo, int(o_id)*7 + 1, 0).r,
                           texelFetch(texSphereInfo, int(o_id)*7 + 2, 0).r);
    this_color = vec3(1,.74,0);
    vec3 this_origin = vec3(texelFetch(texSphereInfo, int(o_id)*7 + 3, 0).r,
                            texelFetch(texSphereInfo, int(o_id)*7 + 4, 0).r,
                            texelFetch(texSphereInfo, int(o_id)*7 + 5, 0).r);

    float this_alpha = texelFetch(texSphereInfo, int(o_id)*7 + 6, 0).r;


    vec3 N_s = normalize(mv_matrix*vec4(normalize(o_pos - this_origin), 0.f)).xyz;
    
    vec3 view =  -(mv_matrix * vec4(o_pos, 1.f)).xyz;
    vec3 lightDir = normalize((mv_matrix*vec4(lightpos,1.f)).xyz + view);
    float NdotLS=dot((N_s),(lightDir));
    vec3 H_S=normalize((lightDir)+normalize(view));

    float specS = dot(H_S,normalize(N_s));
    vec3 diffuse_n_ambient = NdotLS*lightcolor.xyz*this_color*_diffuse + 
     lightcolor.xyz*this_color*_ambient;
    diffuse_n_ambient = max(diffuse_n_ambient, 0.4*this_color);
    diffuse_n_ambient = (diffuse_n_ambient);

    color.xyz = pow(specS,32.f)*lightcolor.xyz*_specular + diffuse_n_ambient;
    
    color.w  = this_alpha;

    return;

}