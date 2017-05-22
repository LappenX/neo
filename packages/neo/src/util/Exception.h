#ifndef NEO_EXCEPTION_H
#define NEO_EXCEPTION_H

#include <string>

#define RETHROW(CATCH, THROW, ...) \
  try \
  { \
    __VA_ARGS__; \
  } \
  catch (const CATCH& e) \
  { \
    throw THROW(e); \
  }

class Exception : public std::exception
{
public:
  Exception(std::string message)
    : m_message(message)
  {
  }

  Exception(std::string message, const Exception& cause)
    : m_message(message + "\nCause: " + std::string(cause.what()))
  {
  }

  virtual ~Exception()
  {
  }

  // TODO: remove all Exception.copy() subclass methods
  const char* what() const throw()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

#endif // NEO_EXCEPTION_H