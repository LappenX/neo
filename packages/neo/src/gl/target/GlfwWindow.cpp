#include "GlfwWindow.h"

#include <GLFW/glfw3.h>

#include <unordered_map>
#include <mutex>

namespace gl {

static std::unordered_map<GLFWwindow*, GlfwWindow*> instances;
static std::mutex instances_mutex;

void key_callback(GLFWwindow* handle, int key, int scancode, int action, int mods)
{
  if (action == GLFW_PRESS)
  {
    if (key != GLFW_KEY_UNKNOWN)
    {
      instances[handle]->m_key_press_event.raise(GlfwKey(key, GLFW_TYPE_KEYBOARD_KEY), true);
    }
    else
    {
      instances[handle]->m_key_press_event.raise(GlfwKey(scancode, GLFW_TYPE_KEYBOARD_SCANCODE), true);
    }
  }
  else if (action == GLFW_RELEASE)
  {
    if (key != GLFW_KEY_UNKNOWN)
    {
      instances[handle]->m_key_press_event.raise(GlfwKey(key, GLFW_TYPE_KEYBOARD_KEY), false);
    }
    else
    {
      instances[handle]->m_key_press_event.raise(GlfwKey(scancode, GLFW_TYPE_KEYBOARD_SCANCODE), false);
    }
  }
}

void cursor_position_callback(GLFWwindow* handle, double x, double y)
{
  instances[handle]->m_cursor_move_event.raise(tensor::Vector2d(x, y));
}

void mouse_button_callback(GLFWwindow* handle, int button, int action, int mods)
{
  if (action == GLFW_PRESS)
  {
    instances[handle]->m_key_press_event.raise(GlfwKey(button, GLFW_TYPE_MOUSE_BUTTON), true);
  }
  else if (action == GLFW_RELEASE)
  {
    instances[handle]->m_key_press_event.raise(GlfwKey(button, GLFW_TYPE_MOUSE_BUTTON), false);
  }
}

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

  glfwSetKeyCallback(m_handle, key_callback);
  glfwSetCursorPosCallback(m_handle, cursor_position_callback);
  glfwSetMouseButtonCallback(m_handle, mouse_button_callback);

  glfwMakeContextCurrent(m_handle);
  glfwSwapInterval(m_vsync ? 1 : 0);
  glfwShowWindow(m_handle);

  instances_mutex.lock();
  instances[m_handle] = this;
  instances_mutex.unlock();

  LOG(info, "gl") << "Created glfw window with title '" << m_title << "'";

  // TODO: char* version = static_cast<const char*>(glGetString(GL_VERSION)); https://www.khronos.org/opengles/sdk/docs/man/xhtml/glGetString.xml
}

GlfwWindow::~GlfwWindow()
{
  glfwDestroyWindow(m_handle);
  instances_mutex.lock();
  instances.erase(m_handle);
  instances_mutex.unlock();
  m_handle = 0;
}

} // end of ns gl