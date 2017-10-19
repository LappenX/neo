#ifndef VIEW_MVC_H
#define VIEW_MVC_H

#include <Common.h>

#include <observer/ObservableEvent.h>
#include <tensor/Tensor.h>

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

  friend KeyboardLayout makeLayoutUS(); // TODO: remove? should always be loaded from file

private:
  KeyCaption m_map[KEY_CAPTION_NUM];
};

inline KeyboardLayout makeLayoutUS() // TODO: remove? see above
{
  KeyboardLayout result;
  for (uint8_t c = 0; c < KEY_CAPTION_NUM; c++)
  {
    result.m_map[c] = static_cast<KeyCaption>(c);
  }
  return result;
}



template <typename TKeyType>
class Controller
{
public:
  virtual void poll() = 0;

  ObservableEvent<TKeyType, bool>& getKeyPressEvent()
  {
    return m_key_press_event;
  }

  ObservableEvent<tensor::Vector2d>& getCursorMoveEvent()
  {
    return m_cursor_move_event;
  }

  virtual bool isKeyDown(const TKeyType& key) = 0;

  virtual void setCursorPosition(tensor::Vector2d pos) = 0;

protected:
  RaisableObservableEvent<TKeyType, bool> m_key_press_event;
  RaisableObservableEvent<tensor::Vector2d> m_cursor_move_event; // TODO: Tensor, and double xpos, ypos; glfwGetCursorPos(window, &xpos, &ypos);
};

#endif // VIEW_MVC_H