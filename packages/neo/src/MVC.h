#ifndef VIEW_MVC_H
#define VIEW_MVC_H

#include <Common.h>

#include <observer/ObservableEvent.h>

class View
{
public:
  virtual void swap() = 0;
};







enum KeyCaption : uint8_t
{
  KEYBOARD_ARROW_UP,
  KEYBOARD_ARROW_DOWN,
  KEYBOARD_ARROW_RIGHT,
  KEYBOARD_ARROW_LEFT,

  MOUSE_LEFT,
  MOUSE_RIGHT,

  KEY_CAPTION_NUM
};

class KeyboardLayout
{
public:
  KeyCaption fromUS(KeyCaption us) const
  {
    return m_map[us];
  }

  KeyCaption& operator[](KeyCaption us)
  {
    return m_map[us];
  }

private:
  KeyCaption m_map[KEY_CAPTION_NUM];
};

template <typename TKeyType>
class Controller
{
public:
  virtual void poll() = 0;

  ObservableEvent<TKeyType, bool>& getKeyPressEvent()
  {
    return m_key_press_event;
  }

  ObservableEvent<double, double>& getCursorMoveEvent()
  {
    return m_cursor_move_event;
  }

  virtual bool isKeyDown(const TKeyType& key) = 0;

  virtual void setCursorPosition(double x, double y) = 0; // TODO: Tensor

protected:
  RaisableObservableEvent<TKeyType, bool> m_key_press_event;
  RaisableObservableEvent<double, double> m_cursor_move_event; // TODO: Tensor, and double xpos, ypos; glfwGetCursorPos(window, &xpos, &ypos);
};

#endif // VIEW_MVC_H