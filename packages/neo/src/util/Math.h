#ifndef NEO_MATH_H
#define NEO_MATH_H

#include <Common.h>
#include <util/Util.h>

#include <cmath>

namespace math {

template <typename T>
struct Consts
{
  static T one;
};

#define OPERATION_T(NAME, EXPRESSION) \
  template <typename T> \
  __host__ __device__ \
  constexpr auto NAME(T x) \
  RETURN_AUTO(EXPRESSION) \
  namespace functor { \
  struct NAME \
  { \
    template <typename T> \
    __host__ __device__ \
    constexpr auto operator()(T x) const \
    RETURN_AUTO(EXPRESSION) \
  }; \
  }

#define OPERATION_TT(NAME, EXPRESSION) \
  template <typename T1, typename T2> \
  __host__ __device__ \
  constexpr auto NAME(T1 x1, T2 x2) \
  RETURN_AUTO(EXPRESSION) \
  namespace functor { \
  struct NAME \
  { \
    template <typename T1, typename T2> \
    __host__ __device__ \
    constexpr auto operator()(T1 x1, T2 x2) const \
    RETURN_AUTO(EXPRESSION) \
  }; \
  }

#define OPERATION_V2(NAME, EXPRESSION_1, EXPRESSION_2, EXPRESSION_N) \
  template <typename T> \
  __host__ __device__ \
  constexpr auto NAME(T x) \
  RETURN_AUTO(EXPRESSION_1) \
  template <typename T1, typename T2> \
  __host__ __device__ \
  constexpr auto NAME(T1 x1, T2 x2) \
  RETURN_AUTO(EXPRESSION_2) \
  template <typename T1, typename T2, typename T3, typename... TRest> \
  __host__ __device__ \
  constexpr auto NAME(T1 x1, T2 x2, T3 x3, TRest... rest) \
  RETURN_AUTO(EXPRESSION_N) \
  namespace functor { \
  struct NAME \
  { \
    template <typename... TTypes> \
    __host__ __device__ \
    constexpr auto operator()(TTypes... xs) const \
    RETURN_AUTO(math:: NAME(xs...)) \
  }; \
  }

#define OPERATION_V1(NAME, EXPRESSION_1, EXPRESSION_N) \
  template <typename T> \
  __host__ __device__ \
  constexpr auto NAME(T x) \
  RETURN_AUTO(EXPRESSION_1) \
  template <typename T, typename... TRest> \
  __host__ __device__ \
  constexpr T NAME(T x1, T x2, TRest... rest) \
  {return (EXPRESSION_N);} \
  namespace functor { \
  struct NAME \
  { \
    template <typename... TTypes> \
    __host__ __device__ \
    constexpr auto operator()(TTypes... xs) const \
    RETURN_AUTO(math:: NAME(xs...)) \
  }; \
  }

OPERATION_V2(add, x, x1 + x2, x1 + math::add(x2, x3, rest...))
OPERATION_TT(subtract, x1 - x2)
OPERATION_V2(multiply, x, x1 * x2, x1 * math::multiply(x2, x3, rest...))
OPERATION_TT(divide, x1 / x2)
OPERATION_T(negate, -x)

OPERATION_T(id, x)
OPERATION_T(abs, x >= 0 ? x : -x)
OPERATION_TT(eq, (x1 == x2))
OPERATION_TT(neq, (x1 != x2))
OPERATION_TT(lt, (x1 < x2))
OPERATION_TT(lte, (x1 <= x2))
OPERATION_TT(gt, (x1 > x2))
OPERATION_TT(gte, (x1 >= x2))
OPERATION_V1(min, x, (x1 < x2 ? math::min(x1, rest...) : math::min(x2, rest...)))
OPERATION_V1(max, x, (x1 > x2 ? math::max(x1, rest...) : math::max(x2, rest...)))

OPERATION_T(sqrt, (T) ::sqrt(x))
OPERATION_T(ln, (T) ::log(x))
OPERATION_T(exp, (T) ::exp(x))

OPERATION_T(x_times_1_minus_x, (T) (x * (1 - x)))
OPERATION_T(sigmoid, (T) (1 / (1 + math::exp(-x))))
OPERATION_T(sigmoid_derivative, math::x_times_1_minus_x(math::sigmoid(x)))

OPERATION_TT(cross_entropy, -(x2 * math::ln(x1) + (1 - x2) * math::ln(1 - x1)))
OPERATION_TT(cross_entropy_derivative, (x1 - x2) / (x1 * (1 - x1)))

} // end of ns math

#endif // NEO_MATH_H