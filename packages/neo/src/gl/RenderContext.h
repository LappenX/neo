#ifndef VIEW_GL_RENDER_CONTEXT_H
#define VIEW_GL_RENDER_CONTEXT_H

#include <Common.h>

#include <util/FastStack.h>
#include "GlError.h"

namespace gl {

const size_t RENDER_STACK_SIZE = 128;

class RenderTarget;
class Shader;

class RenderContext
{
public:
  /*RenderTarget* getActiveTarget()
  {
    if (targets.isEmpty())
    {
      throw RenderException("No render target given");
    }
    return targets.peek();
    // TODO:
    return 0;
  }*/

  Shader* active_shader = 0;
  //FastStack<RENDER_STACK_SIZE, RenderTarget*> targets;
  //FastStack<RENDER_STACK_SIZE, glm::mat4> projection_matrices;
};

} // end of ns gl

#include "target/RenderTarget.h"
#include "core/Shader.h"

#endif // VIEW_GL_RENDER_CONTEXT_H