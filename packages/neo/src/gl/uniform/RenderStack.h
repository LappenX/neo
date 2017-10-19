#ifndef VIEW_GL_UNIFORM_UNIFORMSTACK_H
#define VIEW_GL_UNIFORM_UNIFORMSTACK_H

#include <Common.h>

#include <util/FastStack.h>
#include <util/Math.h>
#include <util/Assert.h>


// TODO: move to different folder
namespace gl {

template <typename T, size_t TSize, typename CRTP>
class RenderStack : public property::Property<const T&>
{
public:
  RenderStack(T empty_value)
    : m_empty_value(empty_value)
  {
  }

  void push(const T& value)
  {
    m_stack.push(static_cast<CRTP*>(this)->process(value));
  }

  void pop()
  {
    m_stack.pop();
  }

  const T& peek() const
  {
    if (m_stack.isEmpty())
    {
      return m_empty_value;
    }
    else
    {
      return m_stack.peek();
    }
  }

  bool isEmpty() const
  {
    return m_stack.isEmpty();
  }

  const T& get() const
  {
    return peek();
  }

  // T process(const T& pushed)

private:
  FastStack<TSize, T> m_stack;
  T m_empty_value;

  NO_COPYING(RenderStack, <T, TSize, CRTP>)
};





template <typename T, size_t TSize>
class MultiplicationRenderStack : public RenderStack<T, TSize, MultiplicationRenderStack<T, TSize>>
{
public:
  MultiplicationRenderStack()
    : RenderStack<T, TSize, MultiplicationRenderStack<T, TSize>>(math::consts::one<T>::get())
  {
  }

  T process(const T& pushed)
  {
    return this->isEmpty() ? pushed : this->peek() * pushed;
  }
};

} // gl

#endif // VIEW_GL_UNIFORM_UNIFORMSTACK_H