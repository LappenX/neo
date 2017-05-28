#include "GlLibrary.h"

#include <util/Logging.h>
#include "GlError.h"

#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <GL/gl.h>

#include <util/Logging.h>

namespace gl {

void glfwErrorCallback(int error, const char* description)
{
  LOG(fatal, "gl") << "Glfw error (" << error << "):" << description << "\n";
  exit(EXIT_FAILURE);
}

namespace glfw {

void init()
{
  glfwSetErrorCallback(glfwErrorCallback);
  if (!glfwInit())
  {
    throw GlException("Failed to initialize GLFW");
  }

  const char* glfw_version = glfwGetVersionString();
  LOG(info, "gl") << "Initialized GLFW version: " << glfw_version;
}

void deinit()
{
  glfwTerminate();
  LOG(info, "gl") << "Terminated GLFW";
}

} // end of ns glfw

namespace glew {

void init()
{
  GLenum code = glewInit();
  if (code != GLEW_OK)
  {
    throw GlException("Failed to initialize GLEW: " + std::string(reinterpret_cast<const char*>(glewGetErrorString(code))));
  }

  LOG(info, "gl") << "Initialized GLEW version: " << glewGetString(GLEW_VERSION);
}

void deinit()
{
}

} // end of ns glew

} // end of ns gl