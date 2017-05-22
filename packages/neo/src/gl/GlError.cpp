#include "GlError.h"

namespace gl {

#ifndef DEBUG

void checkGlError()
{
  GLenum code = glGetError();
  if (code != GL_NO_ERROR)
  {
    throw GlException(code);
  }
}

#else

void checkGlError(std::string description)
{
  GLenum code = glGetError();
  if (code != GL_NO_ERROR)
  {
    throw GlException(description, code);
  }
}

#endif

} // gl