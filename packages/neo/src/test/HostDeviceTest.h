#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <Common.h>

#include <util/Assert.h>

#ifdef __CUDACC__

#define TEST_CASE(NAME) \
  __global__ void kernel_##NAME##_test(); \
  BOOST_AUTO_TEST_CASE(NAME) \
  { \
    CUDA_SAFE_CALL(kernel_##NAME##_test<<<1, 1>>>()); \
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); \
  } \
  __global__ void kernel_##NAME##_test()

#define CHECK(...) ASSERT(__VA_ARGS__, "CUDA device test failed: " << #__VA_ARGS__)

#else

#define TEST_CASE(NAME) BOOST_AUTO_TEST_CASE(NAME)
#define CHECK(...) BOOST_CHECK(__VA_ARGS__)

#endif