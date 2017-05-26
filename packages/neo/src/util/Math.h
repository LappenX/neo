#ifndef NEO_MATH_H
#define NEO_MATH_H

#include <Common.h>
#include <util/Util.h>

#include <cmath>

namespace math {

namespace consts {

template <typename T>
struct one;

} // end of ns consts

#define TYPE_POSTFIX &&


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
  constexpr auto NAME(T1 TYPE_POSTFIX x1, T2 TYPE_POSTFIX x2) \
  RETURN_AUTO(EXPRESSION) \
  namespace functor { \
  struct NAME \
  { \
    template <typename T1, typename T2> \
    __host__ __device__ \
    constexpr auto operator()(T1&& x1, T2&& x2) const \
    RETURN_AUTO(math:: NAME(util::forward<T1>(x1), util::forward<T2>(x2))) \
  }; \
  }
// TODO: no type in OPERATION_V2 EXPRESSION_0 function?
#define OPERATION_V2(NAME, EXPRESSION_0, EXPRESSION_1, EXPRESSION_N) \
  namespace detail { \
  struct NAME##Helper { \
  __host__ __device__ \
  static constexpr auto NAME() \
  RETURN_AUTO(EXPRESSION_0) \
  template <typename T> \
  __host__ __device__ \
  static constexpr auto NAME(T x) \
  RETURN_AUTO(EXPRESSION_1) \
  template <typename T1, typename T2, typename... TRest> \
  __host__ __device__ \
  static constexpr auto NAME(T1 TYPE_POSTFIX x1, T2 TYPE_POSTFIX x2, TRest TYPE_POSTFIX ... rest) \
  RETURN_AUTO(EXPRESSION_N) \
  }; \
  } \
  template <typename... TTypes> \
  __host__ __device__ \
  constexpr auto NAME(TTypes&&... xs) \
  RETURN_AUTO(detail:: NAME##Helper :: NAME(util::forward<TTypes>(xs)...)) \
  namespace functor { \
  struct NAME \
  { \
    template <typename... TTypes> \
    __host__ __device__ \
    constexpr auto operator()(TTypes&&... xs) const \
    RETURN_AUTO(math:: NAME(util::forward<TTypes>(xs)...)) \
  }; \
  }

#define OPERATION_V1(NAME, EXPRESSION_1, EXPRESSION_N) \
  namespace detail { \
  struct NAME##Helper { \
  template <typename T> \
  __host__ __device__ \
  static constexpr T NAME(T x) \
  {return (EXPRESSION_1);} \
  template <typename T, typename... TRest> \
  __host__ __device__ \
  static constexpr T NAME(T x1, T x2, TRest... rest) \
  {return (EXPRESSION_N);} \
  }; \
  } \
  template <typename... TTypes> \
  __host__ __device__ \
  constexpr auto NAME(TTypes&&... xs) \
  RETURN_AUTO(detail:: NAME##Helper :: NAME(util::forward<TTypes>(xs)...)) \
  namespace functor { \
  struct NAME \
  { \
    template <typename... TTypes> \
    __host__ __device__ \
    constexpr auto operator()(TTypes&&... xs) const \
    RETURN_AUTO(math:: NAME(util::forward<TTypes>(xs)...)) \
  }; \
  }



#define X1 util::forward<T1>(x1)
#define X2 util::forward<T2>(x2)
#define REST util::forward<TRest>(rest)

OPERATION_V2(add, 0, x, add(X1 + X2, REST...))
OPERATION_TT(subtract, X1 - X2)
OPERATION_V2(multiply, 1, x, X1 * multiply(X2, REST...))
OPERATION_TT(divide, X1 / X2)
OPERATION_T(negate, -x)

OPERATION_T(id, x)
OPERATION_T(abs, x >= 0 ? x : -x)
OPERATION_TT(eq, (X1 == X2))
OPERATION_TT(neq, (X1 != X2))
OPERATION_TT(lt, (X1 < X2))
OPERATION_TT(lte, (X1 <= X2))
OPERATION_TT(gt, (X1 > X2))
OPERATION_TT(gte, (X1 >= X2))

OPERATION_V1(min, x, (x1 < x2 ? min(x1, rest...) : min(x2, rest...)))
OPERATION_V1(max, x, (x1 > x2 ? max(x1, rest...) : max(x2, rest...)))

OPERATION_V2(land, true, x, land(X1 && X2, REST...))
OPERATION_V2(lor, false, x, lor(X1 || X2, REST...))

OPERATION_T(sqrt, (T) ::sqrt(x))
OPERATION_T(ln, (T) ::log(x))
OPERATION_T(exp, (T) ::exp(x))
OPERATION_T(tan, (T) ::tan(x))

OPERATION_T(x_times_1_minus_x, (T) (x * (1 - x)))
OPERATION_T(sigmoid, (T) (1 / (1 + math::exp(-x))))
OPERATION_T(sigmoid_derivative, math::x_times_1_minus_x(math::sigmoid(x)))

OPERATION_TT(cross_entropy, -(X2 * math::ln(X1) + (1 - X2) * math::ln(1 - X1)))
OPERATION_TT(cross_entropy_derivative, (X1 - X2) / (X1 * (1 - X1)))

#undef X1
#undef X2
#undef REST

} // end of ns math

#endif // NEO_MATH_H