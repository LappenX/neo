#ifndef VIEW_GL_TARGET_RENDERTARGET_H
#define VIEW_GL_TARGET_RENDERTARGET_H

#include <Common.h>

namespace gl {

class Shader;

class RenderTarget
{
public:
  virtual void bind() = 0;
};

} // end of ns gl

#include "../core/Shader.h"

#endif // VIEW_GL_TARGET_RENDERTARGET_H