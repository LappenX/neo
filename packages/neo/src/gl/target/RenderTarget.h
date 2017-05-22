#ifndef VIEW_GL_TARGET_RENDERTARGET_H
#define VIEW_GL_TARGET_RENDERTARGET_H

#include <Common.h>
#include "../RenderStep.h"

namespace gl {

class RenderContext;
class Shader;

class RenderTarget : public UnRenderStep
{
public:
  virtual void bind() = 0;

  void render(RenderContext& context)
  {
    this->bind();
  }
};

} // end of ns gl

#include "../core/Shader.h"

#endif // VIEW_GL_TARGET_RENDERTARGET_H