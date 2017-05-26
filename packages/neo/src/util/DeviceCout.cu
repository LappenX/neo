#include "DeviceCout.h"

#include <stdio.h>

namespace util {

namespace detail {

#define PRINT_TYPE(TYPE, TOKEN) \
  __device__ \
  void printDevice(TYPE object) \
  { \
    printf(TOKEN, object); \
  }

PRINT_TYPE(const char*, "%s")
PRINT_TYPE(unsigned char, "%hhu")
PRINT_TYPE(unsigned short int, "%hu")
PRINT_TYPE(unsigned int, "%u")
PRINT_TYPE(unsigned long int, "%lu")
PRINT_TYPE(unsigned long long int, "%llu")
PRINT_TYPE(char, "%hhd")
PRINT_TYPE(short int, "%hd")
PRINT_TYPE(int, "%d")
PRINT_TYPE(long int, "%ld")
PRINT_TYPE(long long int, "%lld")
PRINT_TYPE(float, "%f")
PRINT_TYPE(double, "%f")

__device__
void printDevice(bool object)
{
  if (object)
  {
    printf("true");
  }
  else
  {
    printf("false");
  }
}

#undef PRINT_TYPE

} // end of ns detail

} // end of ns util