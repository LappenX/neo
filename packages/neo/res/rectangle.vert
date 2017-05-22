#version 150

uniform mat4 uModelViewProjectionMatrix;

in vec3 in_location;
//in vec2 in_tex_coord;

//out FRAG_IN {
//    vec2 tex_coord;
//} vert_out;

void main() {
    gl_Position = uModelViewProjectionMatrix * vec4(in_location, 1.0);
    //vert_out.tex_coord = in_tex_coord;
}

