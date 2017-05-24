#ifndef NEO_ASSERTION_H
#define NEO_ASSERTION_H

#include <Common.h>



#ifdef DEBUG

#ifdef __CUDA_ARCH__
#define EXIT asm("trap;")
#else
#define EXIT exit(EXIT_FAILURE)
#endif

#ifndef __CUDA_ARCH__
#include <iostream>
#endif





namespace detail {

class AssertCout
{
public:
  template <typename T>
  __host__ __device__
  AssertCout& operator<<(T object);
};

#ifndef __CUDA_ARCH__
template <typename T>
__host__ __device__
AssertCout& AssertCout::operator<<(T object)
{
  std::cout << object;
}
#endif

} // end of ns detail





#define ASSERT(cond, ...) \
  do \
  { \
    if (!(cond)) \
    { \
      ::detail::AssertCout() << "\nAssertion '" << #cond << "' failed in " << __FILE__ << ":" << __LINE__ << "!\n" << __VA_ARGS__ << "\n"; \
      EXIT; \
    } \
  } while(0)

#define CUDA_SAFE_CALL(...) \
    do \
    { \
      __VA_ARGS__; \
      cudaError_t err = cudaGetLastError(); \
      if (err != cudaSuccess) \
      { \
        ::detail::AssertCout() << "\nCuda safe call '" << #__VA_ARGS__ << "' failed in " << __FILE__ << ":" << __LINE__ << "!\nErrCode" << err << "\n" << cudaGetErrorString(err); \
        EXIT; \
      } \
    } while(false)



#else
#define ASSERT(cond, ...)
#define CUDA_SAFE_CALL(...) __VA_ARGS__
#endif

#endif // NEO_ASSERTION_H
