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
  virtual void bind() = 0; // TODO: bind in pre
  virtual void afterDraw() = 0; // TODO: somewhere global? So that swapping doesnt happen more than once per loop, remove afterDraw in RenderTarget

  void render(RenderContext& context)
  {
    this->bind();
  }
};

} // end of ns gl

#include "../core/Shader.h"

#endif // VIEW_GL_TARGET_RENDERTARGET_H