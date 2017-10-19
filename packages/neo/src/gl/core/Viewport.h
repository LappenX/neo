#ifndef VIEW_GL_CORE_VIEWPORT_H
#define VIEW_GL_CORE_VIEWPORT_H

#include <Common.h>

#include <GL/gl.h>

namespace gl {

class Viewport
{
public:
  Viewport(size_t x, size_t y, size_t width, size_t height)
    : m_x(x)
    , m_y(y)
    , m_width(width)
    , m_height(height)
  {
  }

  void set()
  {
    glViewport(m_x, m_y, m_width, m_height);
  }

private:
  size_t m_x;
  size_t m_y;
  size_t m_width;
  size_t m_height;
};

} // gl

#endif // VIEW_GL_CORE_VIEWPORT_H