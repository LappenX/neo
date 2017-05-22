#ifndef VIEW_GL_TARGET_FRAMEBUFFEROBJECT
#define VIEW_GL_TARGET_FRAMEBUFFEROBJECT

#include <Common.h>

#include <GL/gl.h>

namespace gl {

class FrameBufferObject : public RenderTarget
{
public:
  void bind()
  {
    // TODO: setDrawBuffers:
    /*
  IntBuffer buffer = BufferUtils.createIntBuffer(shader.getFragOutputs().size());
        for (String fragOutput : shader.getFragOutputs()) {
            Integer attachment = colorAttachments.get(fragOutput);
            if (attachment != null) {
                buffer.put(attachment);
            } else {
                buffer.put(GL11.GL_NONE);
            }
        }
        buffer.flip();
        GL20.glDrawBuffers(buffer);
OpenGLException.checkGLError("Failed to set draw buffers for framebuffer object");
    */
  }
};

} // gl

#endif // VIEW_GL_TARGET_FRAMEBUFFEROBJECT