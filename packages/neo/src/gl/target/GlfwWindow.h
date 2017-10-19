#ifndef VIEW_GL_GLFWWINDOW_H
#define VIEW_GL_GLFWWINDOW_H

#include <Common.h>

#include "../target/RenderTarget.h"
#include <MVC.h>
#include <tensor/Tensor.h>

#include <string>
#include <GLFW/glfw3.h>

namespace gl {

class GlfwWindow;

enum GlfwKeyType
{
  GLFW_TYPE_KEYBOARD_KEY,
  GLFW_TYPE_KEYBOARD_SCANCODE,
  GLFW_TYPE_MOUSE_BUTTON
};

class GlfwKey
{
public:
  GlfwKey(int code, GlfwKeyType type)
    : m_code(code)
    , m_type(type)
  {
  }

  KeyCaption getKeyCaption(const KeyboardLayout& layout) const
  {
    return layout.fromUS(getUSKeyCaption());
  }

  friend class GlfwWindow;

private:
  int m_code;
  GlfwKeyType m_type;

  KeyCaption getUSKeyCaption() const
  {
    switch (m_type)
    {
      case GLFW_TYPE_KEYBOARD_KEY:
      {
        switch (m_code)
        {
          case GLFW_KEY_LEFT: return KEYBOARD_ARROW_LEFT;
          case GLFW_KEY_RIGHT: return KEYBOARD_ARROW_RIGHT;
          case GLFW_KEY_UP: return KEYBOARD_ARROW_UP;
          case GLFW_KEY_DOWN: return KEYBOARD_ARROW_DOWN;
          default:
          {
            throw GlException("No caption for this key");
          }
        }
      }
      case GLFW_TYPE_KEYBOARD_SCANCODE:
      {
        throw GlException("Cannot get caption for a key's scancode");
      }
      case GLFW_TYPE_MOUSE_BUTTON:
      {
        switch (m_code)
        {
          case GLFW_MOUSE_BUTTON_LEFT: return MOUSE_LEFT;
          case GLFW_MOUSE_BUTTON_RIGHT: return MOUSE_RIGHT;
          default:
          {
            throw GlException("No caption for this key");
          }
        }
      }
      default:
      {
        ASSERT(false, "Invalid key type");
        return KEY_CAPTION_NUM;
      }
    }
  }
};

class GlfwWindow : public RenderTarget, public View, public Controller<GlfwKey>
{
public:
  GlfwWindow(std::string title, uint32_t width, uint32_t height, uint32_t refresh_rate, bool vsync);
  virtual ~GlfwWindow();

  uint32_t getWidth() const
  {
    return m_width;
  }

  uint32_t getHeight() const
  {
    return m_height;
  }

  void bind()
  {
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // TODO: Cp. FrameBufferObject.h
  }

  void swap()
  {
    glfwSwapBuffers(m_handle);
  }

  void poll()
  {
    glfwPollEvents();
  }

  bool isKeyDown(const GlfwKey& key)
  {
    switch (key.m_type)
    {
      case GLFW_TYPE_KEYBOARD_KEY:
      {
        return glfwGetKey(m_handle, key.m_code) == GLFW_PRESS;
      }
      case GLFW_TYPE_KEYBOARD_SCANCODE:
      {
        throw GlException("Cannot call glfwGetKey for a key's scancode");
      }
      case GLFW_TYPE_MOUSE_BUTTON:
      {
        return glfwGetMouseButton(m_handle, key.m_code) == GLFW_PRESS;
      }
      default:
      {
        ASSERT(false, "Invalid key type");
        return false;
      }
    }
  }

  void setCursorPosition(tensor::Vector2d pos)
  {
    glfwSetCursorPos(m_handle, pos(0), pos(1));
  }

  tensor::Vector2ui getSize() const
  {
    return tensor::Vector2ui(m_width, m_height); // TODO: store as tensor in class?
  }

  friend void key_callback(GLFWwindow* handle, int key, int scancode, int action, int mods);
  friend void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
  friend void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

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