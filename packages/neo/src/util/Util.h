#ifndef NEO_UTIL_H
#define NEO_UTIL_H

#include <Common.h>

#include <utility>
#include <type_traits>


#define NO_COPYING(CLASS, ...)  CLASS __VA_ARGS__ & operator=(const CLASS __VA_ARGS__ &) = delete; \
                                        CLASS(const CLASS __VA_ARGS__ &) = delete;
#define TVALUE(TYPE, NAME, ...) struct NAME {static const TYPE value = __VA_ARGS__;};
#define RETURN_AUTO(...) -> decltype(__VA_ARGS__) {return __VA_ARGS__;}


namespace util {

template <typename T>
__host__ __device__
T&& forward(typename std::remove_reference<T>::type& t) noexcept
{
  return static_cast<T&&>(t);
}

template <typename T>
__host__ __device__
T&& forward(typename std::remove_reference<T>::type&& t) noexcept
{
  static_assert(!std::is_lvalue_reference<T>::value, "Cannot forward an rvalue as an lvalue.");
  return static_cast<T&&>(t);
}

template <typename T>
__host__ __device__
typename std::remove_reference<T>::type&& move(T&& t)
{
  return static_cast<typename std::remove_reference<T>::type&&>(t);
}

} // end of ns util

#endif // NEO_UTIL_H
