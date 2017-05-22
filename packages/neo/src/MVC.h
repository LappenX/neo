#ifndef VIEW_MVC_H
#define VIEW_MVC_H

#include <Common.h>

#include <observer/ObservableEvent.h>

class View
{
public:
  virtual void swap() = 0;
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