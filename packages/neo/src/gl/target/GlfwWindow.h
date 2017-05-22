#ifndef VIEW_GL_GLFWWINDOW_H
#define VIEW_GL_GLFWWINDOW_H

#include <Common.h>

#include "../target/RenderTarget.h"

#include <string>
#include <GLFW/glfw3.h>

namespace gl {

class GlfwWindow : public RenderTarget
{
public:
  GlfwWindow(std::string title, uint32_t width, uint32_t height, uint32_t refresh_rate, bool vsync);
  virtual ~GlfwWindow();

  void bind()
  {
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // TODO: Cp. FrameBufferObject.h
  }

  void afterDraw()
  {
    glfwSwapBuffers(m_handle);
    glfwPollEvents();
  }

private:
  std::string m_title;
  uint32_t m_width;
  uint32_t m_height;
  GLFWwindow* m_handle;
  uint32_t m_refresh_rate;
  bool m_vsync;
};

} // end of ns gl

#endif // VIEW_GL_GLFWWINDOW_H