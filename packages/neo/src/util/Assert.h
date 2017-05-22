#ifndef NEO_ASSERTION_H
#define NEO_ASSERTION_H

#include <Common.h>

#ifdef DEBUG

#include <string>

// TODO: differentiate between cuda host and device
namespace detail {

void fail_assertion(const char* cond_name, const char* file, uint32_t line, const char* message);
void fail_assertion(const char* cond_name, const char* file, uint32_t line, std::string message);

} // end of ns detail

#define ASSERT(cond, message) \
  do \
  { \
    if (!(cond)) \
    { \
      ::detail::fail_assertion(#cond, __FILE__, __LINE__, message); \
    } \
  } while(0)
#else
#define ASSERT(cond, message)
#endif

#endif // NEO_ASSERTION_H
