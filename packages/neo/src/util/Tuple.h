#ifndef NEO_TUPLE_H
#define NEO_TUPLE_H

#include <Common.h>
#include <util/Util.h>

namespace tuple {

template <typename... TTypes>
class Tuple;



namespace detail {

template <size_t I, typename... TTypes>
struct TupleGet;

template <size_t I, typename TFirst, typename... TRest>
struct TupleGet<I, TFirst, TRest...>
{
  using output_t = typename TupleGet<I - 1, TRest...>::output_t;

  __host__ __device__
  static output_t& get(Tuple<TFirst, TRest...>& tuple)
  {
    return TupleGet<I - 1, TRest...>::get(tuple.m_rest);
  }

  __host__ __device__
  static const output_t& get(const Tuple<TFirst, TRest...>& tuple)
  {
    return TupleGet<I - 1, TRest...>::get(tuple.m_rest);
  }
};

template <typename TFirst, typename... TRest>
struct TupleGet<0, TFirst, TRest...>
{
  using output_t = TFirst;

  __host__ __device__
  static output_t& get(Tuple<TFirst, TRest...>& tuple)
  {
    return tuple.m_first;
  }

  __host__ __device__
  static const output_t& get(const Tuple<TFirst, TRest...>& tuple)
  {
    return tuple.m_first;
  }
};

} // end of ns detail






template <typename TFirst, typename... TRest>
class Tuple<TFirst, TRest...>
{
public:
  template <size_t I>
  using nth_type_t = typename detail::TupleGet<I, TFirst, TRest...>::output_t;

  template <typename TFirst2, typename... TRest2>
  __host__ __device__
  Tuple(TFirst2&& first, TRest2&&... rest)
    : m_first(util::forward<TFirst2>(first))
    , m_rest(util::forward<TRest2>(rest)...)
  {
  }


  template <size_t I, typename... TTypes>
  friend struct detail::TupleGet;

  template <size_t I>
  __host__ __device__
  nth_type_t<I>& get()
  {
    return detail::TupleGet<I, TFirst, TRest...>::get(*this);
  }

  template <size_t I>
  __host__ __device__
  const nth_type_t<I>& get() const
  {
    return detail::TupleGet<I, TFirst, TRest...>::get(*this);
  }

  template <typename TFunc>
  void for_each(TFunc functor) const
  {
    functor(m_first);
    m_rest.for_each(functor);
  }

private:
  TFirst m_first;
  Tuple<TRest...> m_rest;
};

template <>
class Tuple<>
{
public:
  template <typename TFunc>
  void for_each(TFunc functor) const
  {
  }
};





template <size_t I>
struct nth_element
{
  template <typename TFirst, typename... TRest>
  __host__ __device__
  static auto get(TFirst&& first, TRest&&... rest)
  RETURN_AUTO(nth_element<I - 1>::get(util::forward<TRest>(rest)...))
};

template <>
struct nth_element<0>
{
  template <typename TFirst, typename... TRest>
  __host__ __device__
  static auto get(TFirst&& first, TRest&&... rest)
  RETURN_AUTO(util::forward<TFirst>(first))
};





template <bool TBool>
struct conditional
{
  template <typename TIfTrue, typename TIfFalse>
  __host__ __device__
  static auto get(TIfTrue&& if_true, TIfFalse&& if_false)
  RETURN_AUTO(util::forward<TIfTrue>(if_true))
};

template <>
struct conditional<false>
{
  template <typename TIfTrue, typename TIfFalse>
  __host__ __device__
  static auto get(TIfTrue&& if_true, TIfFalse&& if_false)
  RETURN_AUTO(util::forward<TIfFalse>(if_false))
};





template <typename TFunc>
__host__ __device__
void for_each(TFunc func)
{
}

template <typename TFunc, typename TArg0>
__host__ __device__
void for_each(TFunc func, TArg0&& arg0)
{
   func(util::forward<TArg0>(arg0));
}

template <typename TFunc, typename TArg0, typename... TArgs>
__host__ __device__
void for_each(TFunc func, TArg0&& arg0, TArgs&&... args)
{
   func(util::forward<TArg0>(arg0));
   for_each(func, util::forward<TArgs>(args)...);
}





template <typename TFunction, typename... TArgs>
using result_type_t = decltype(TFunction(std::declval<TArgs>()...));

} // end of ns tuple

#endif // NEO_TUPLE_H
