#ifndef NEO_UTIL_H
#define NEO_UTIL_H

#include <Common.h>

#include <utility>
#include <memory>
#include <type_traits>

namespace detail {

static const union
{
  uint8_t bytes[4];
  uint32_t value;
} endianness_helper = {{0, 1, 2, 3}};

} // end of ns detail

#define IS_LITTLE_ENDIAN (::detail::endianness_helper.value == 0x03020100ul)
#define IS_BIG_ENDIAN (::detail::endianness_helper.value == 0x00010203ul)








#define NO_COPYING(CLASS, ...)  CLASS __VA_ARGS__ & operator=(const CLASS __VA_ARGS__ &) = delete; \
                                        CLASS(const CLASS __VA_ARGS__ &) = delete;
#define TVALUE(TYPE, NAME, ...) struct NAME {static const TYPE value = __VA_ARGS__;};
#define RETURN_AUTO(...) -> decltype(__VA_ARGS__) {return __VA_ARGS__;}
#define ENABLE_IF(...) typename = typename std::enable_if<__VA_ARGS__, void>::type
#define ENABLE_IF_ARE_SIZE_T(...) ENABLE_IF(tmp::type_sequence::all_apply_v<tmp::type_sequence::pred::is_convertible_to<size_t>, tmp::type_sequence::Sequence<__VA_ARGS__>>::value)
namespace util {struct EmptyDefaultType {};}
#define WITH_DEFAULT_TYPE(TYPE, DEFAULT) typename std::conditional<std::is_same<TYPE, util::EmptyDefaultType>::value, DEFAULT, TYPE>::type
#define NO_DELETION_SHARED_PTR(TYPE, PTR) (std::shared_ptr<TYPE>(std::shared_ptr<TYPE>{}, PTR))

namespace util {

template <typename T>
__host__ __device__
constexpr T&& forward(typename std::remove_reference<T>::type& t) noexcept
{
  return static_cast<T&&>(t);
}

template <typename T>
__host__ __device__
constexpr T&& forward(typename std::remove_reference<T>::type&& t) noexcept
{
  static_assert(!std::is_lvalue_reference<T>::value, "Cannot forward an rvalue as an lvalue.");
  return static_cast<T&&>(t);
}

template <typename T>
__host__ __device__
constexpr typename std::remove_reference<T>::type&& move(T&& t)
{
  return static_cast<typename std::remove_reference<T>::type&&>(t);
}

template <typename T, typename... TArgs>
std::unique_ptr<T> make_unique(TArgs&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<TArgs>(args)...));
}



struct deleter
{
  template <typename T>
  void operator()(T* object)
  {
    delete object;
  }
};

} // end of ns util

#endif // NEO_UTIL_H
