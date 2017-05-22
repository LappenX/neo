#ifndef VIEW_GL_CORE_CLEARBUFFER_H
#define VIEW_GL_CORE_CLEARBUFFER_H

#include <Common.h>

#include "../RenderStep.h"
#include "../Glm.h"
#include "../GlError.h"

#include <GL/glew.h>
#include <GL/gl.h>

namespace gl {

class ClearColorBuffer : public UnRenderStep
{
public:
  ClearColorBuffer(glm::vec4 color)
    : m_color(color)
  {
  }

  void render(RenderContext& context)
  {
    glClearColor(m_color[0], m_color[1], m_color[2], m_color[3]);
    glClear(GL_COLOR_BUFFER_BIT);
    GL_CHECK_ERROR("Failed to clear color buffer");
  }

private:
  glm::vec4 m_color;
};

} // gl

#endif // VIEW_GL_CORE_CLEARBUFFER_H