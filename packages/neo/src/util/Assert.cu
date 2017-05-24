#include "Assert.h"

#ifdef DEBUG

namespace detail {

template <>
__host__ __device__
AssertCout& AssertCout::operator<<(const char* object)
{
  printf("%s", object);
  return *this;
}


/*
if (object)
  {
    printf("true");
  }
  else
  {
    printf("false");
  }
  */

} // end of ns detail

#endif