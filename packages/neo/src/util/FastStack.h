#ifndef NEO_FASTSTACK_H
#define NEO_FASTSTACK_H

#include <Common.h>
#include <util/Assert.h>

template <size_t TSize, typename T>
class FastStack
{
public:
  FastStack()
    : m_pos(0)
  {}

  void push(T obj)
  {
    ASSERT(m_pos < TSize, "Maximum stack size reached");
    stack[m_pos++] = obj;
  }

  T pop()
  {
    ASSERT(!isEmpty(), "Stack is empty");
    return stack[--m_pos];
  }

  const T& peek() const
  {
    ASSERT(!isEmpty(), "Stack is empty");
    return stack[m_pos - 1];
  }

  bool isEmpty() const
  {
    return m_pos == 0;
  }

private:
  size_t m_pos; // Position of next free slot
  T stack[TSize];
};

#endif // NEO_FASTSTACK_H
