#version 150

//uniform sampler2D uTexture;

//in FRAG_IN {
//    vec2 tex_coord;
//} frag_in;

out vec4 out_fragment;

void main() {
    out_fragment = vec4(1.0, 0.0, 0.0, 1.0); //texture2D(uTexture, frag_in.tex_coord);
}

