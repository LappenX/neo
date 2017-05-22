#include "GlfwWindow.h"

#include <GLFW/glfw3.h>

namespace gl {

GlfwWindow::GlfwWindow(std::string title, uint32_t width, uint32_t height, uint32_t refresh_rate, bool vsync)
    : m_title(title)
    , m_width(width)
    , m_height(height)
    , m_handle(0)
    , m_refresh_rate(refresh_rate)
    , m_vsync(vsync)
{
  glfwDefaultWindowHints();
  glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
  glfwWindowHint(GLFW_REFRESH_RATE, m_refresh_rate);

  m_handle = glfwCreateWindow(m_width, m_height, m_title.c_str(), NULL, NULL);
  if (!m_handle)
  {
    throw GlException("Failed to create GLFW window");
  }

  glfwMakeContextCurrent(m_handle);
  glfwSwapInterval(m_vsync ? 1 : 0);
  glfwShowWindow(m_handle);

  LOG(info, "gl") << "Created glfw window with title '" << m_title << "'";

  // TODO: char* version = static_cast<const char*>(glGetString(GL_VERSION)); https://www.khronos.org/opengles/sdk/docs/man/xhtml/glGetString.xml
}

GlfwWindow::~GlfwWindow()
{
  glfwDestroyWindow(m_handle);
  m_handle = 0;
}

} // end of ns gl