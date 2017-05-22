#include "assert.h"

#include <iostream>
#include <cstdlib>

#ifdef DEBUG

namespace detail {

void fail_assertion(const char* cond_name, const char* file, uint32_t line, const char* message)
{ // TODO: boost logging
  std::cerr << std::endl;
  std::cerr << "Assertion '" << cond_name << "' failed in " << file << ":" << line << "!" << std::endl;
  std::cerr << message << std::endl;
  exit(EXIT_FAILURE);
}

void fail_assertion(const char* cond_name, const char* file, uint32_t line, std::string message)
{
  fail_assertion(cond_name, file, line, message.c_str());
}

} // end of ns detail

#endif