#ifndef VIEW_GL_GLEW_H
#define VIEW_GL_GLEW_H

#include <Common.h>

namespace gl {

void glfwErrorCallback(int error, const char* description);

namespace glfw {

void init();
void deinit();

} // end of ns glfw

namespace glew {

void init();
void deinit();

} // end of ns glew

} // end of ns gl

#endif // VIEW_GL_GLEW_H