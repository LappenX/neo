#ifndef DEVICE_COUT_H
#define DEVICE_COUT_H

#include <Common.h>

namespace util {

namespace detail {

#define PRINT_TYPE(TYPE) \
  __device__ \
  void printDevice(TYPE object);

PRINT_TYPE(const char*)
PRINT_TYPE(unsigned char)
PRINT_TYPE(unsigned short int)
PRINT_TYPE(unsigned int)
PRINT_TYPE(unsigned long int)
PRINT_TYPE(unsigned long long int)
PRINT_TYPE(char)
PRINT_TYPE(short int)
PRINT_TYPE(int)
PRINT_TYPE(long int)
PRINT_TYPE(long long int)
PRINT_TYPE(float)
PRINT_TYPE(double)
PRINT_TYPE(bool)

#undef PRINT_TYPE

} // end of ns detail



class DeviceCout
{
public:
  template <typename T>
  __host__ __device__
  DeviceCout& operator<<(T object)
  {
    detail::printDevice(object);
    return *this;
  }
};

} // end of ns util

#endif // DEVICE_COUT_H