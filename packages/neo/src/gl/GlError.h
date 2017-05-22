#ifndef VIEW_GL_GLERROR_H
#define VIEW_GL_GLERROR_H

#include <Common.h>

#include <util/Exception.h>

#include <string>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

namespace gl {

#ifndef DEBUG

class GlException : public Exception
{
public:
  GlException(GLenum code)
    : Exception(reinterpret_cast<const char*>(gluErrorString(code)))
  {
  }

  GlException(const char* message)
    : Exception(message)
  {
  }

  GlException(std::string message)
    : Exception(message)
  {
  }
};

#define GL_CHECK_ERROR(DESCRIPTION) checkGlError()
void checkGlError();

#else

class GlException : public Exception
{
public:
  GlException(std::string description, GLenum code)
    : Exception(description + " (" + std::string(reinterpret_cast<const char*>(gluErrorString(code))) + ")")
  {
  }

  GlException(const char* message)
    : Exception(message)
  {
  }

  GlException(std::string message)
    : Exception(message)
  {
  }
};

#define GL_CHECK_ERROR(DESCRIPTION) checkGlError(DESCRIPTION)
void checkGlError(std::string description);

#endif

} // gl

#endif // VIEW_GL_GLERROR_H







